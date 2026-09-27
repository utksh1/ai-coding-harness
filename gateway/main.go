// Foreman API Gateway (platform P2, issue #71).
//
// TECHNICAL_IMPLEMENTATION.md §2.2 Service 1: HTTP routing, WebSocket
// broadcast of engine events, proxying to the Python orchestrator.
// Endpoints: GET /api/health, GET|POST /api/tasks, GET /api/tasks/:id,
// GET /api/agents, GET /api/metrics, WS /ws.
package main

import (
        "bytes"
        "context"
        "embed"
        "encoding/json"
        "io"
        "log"
        "net/http"
        "net/http/httptest"
        "net/http/httputil"
        "net/url"
        "os"
        "sync"
        "time"

        "github.com/google/uuid"
        "github.com/gorilla/websocket"
        "github.com/redis/go-redis/v9"
)

//go:embed web/*
var webFS embed.FS

// Config from the environment (12-factor; never hard-coded credentials).
type Config struct {
        Addr            string
        OrchestratorURL string
        RedisAddr       string
}

func LoadConfig() Config {
        return Config{
                Addr:            envOr("GATEWAY_ADDR", ":8080"),
                OrchestratorURL: envOr("ORCHESTRATOR_URL", "http://localhost:8000"),
                RedisAddr:       envOr("REDIS_ADDR", "localhost:6379"),
        }
}

func envOr(key, fallback string) string {
        if value := os.Getenv(key); value != "" {
                return value
        }
        return fallback
}

// Hub broadcasts engine events to every connected WebSocket client.
type Hub struct {
        mu      sync.Mutex
        clients map[*websocket.Conn]bool
}

func NewHub() *Hub {
        return &Hub{clients: make(map[*websocket.Conn]bool)}
}

func (h *Hub) add(client *websocket.Conn) {
        h.mu.Lock()
        defer h.mu.Unlock()
        h.clients[client] = true
}

func (h *Hub) remove(client *websocket.Conn) {
        h.mu.Lock()
        defer h.mu.Unlock()
        delete(h.clients, client)
}

func (h *Hub) Broadcast(event []byte) {
        h.mu.Lock()
        defer h.mu.Unlock()
        for client := range h.clients {
                _ = client.WriteMessage(websocket.TextMessage, event)
        }
}

func (h *Hub) Count() int {
        h.mu.Lock()
        defer h.mu.Unlock()
        return len(h.clients)
}

// Gateway wires the handlers to their dependencies (testable without network).
type Gateway struct {
        config      Config
        hub         *Hub
        upstream    *url.URL
        proxy       http.Handler
        taskEvents  map[string][]json.RawMessage // run_id -> events (fallback store)
        taskEventsMu sync.Mutex
}

func NewGateway(config Config, proxy http.Handler) *Gateway {
        return &Gateway{
                config:     config,
                hub:        NewHub(),
                proxy:      proxy,
                taskEvents: make(map[string][]json.RawMessage),
        }
}

// maxStoredEvents caps one run's in-memory event history: the cockpit's
// feed is a live stream, the history is a convenience, and an unbounded map
// is a leak on a long-lived daemon.
const maxStoredEvents = 500

// Handler builds the full route table.
func (g *Gateway) Handler() http.Handler {
        mux := http.NewServeMux()
        mux.HandleFunc("GET /{$}", g.handleDashboard)
        mux.HandleFunc("GET /dashboard", g.handleDashboard)
        mux.HandleFunc("GET /api/health", g.handleHealth)
        mux.HandleFunc("POST /api/events", g.handleBroadcastEvent)
        mux.HandleFunc("POST /api/tasks", g.handleCreateTask)
        mux.HandleFunc("GET /api/tasks/{id}", g.handleTask)
        mux.HandleFunc("GET /api/agents", g.proxyToOrchestrator)
        mux.HandleFunc("GET /api/evidence/{rest...}", g.proxyToOrchestrator)
        mux.HandleFunc("GET /api/metrics", g.handleMetrics)
        mux.HandleFunc("GET /ws", g.handleWS)
        return mux
}

func (g *Gateway) handleDashboard(w http.ResponseWriter, _ *http.Request) {
        content, err := webFS.ReadFile("web/index.html")
        if err != nil {
                content, err = os.ReadFile("web/index.html")
                if err != nil {
                        content, err = os.ReadFile("gateway/web/index.html")
                }
        }
        if err != nil {
                http.Error(w, "dashboard not found", http.StatusNotFound)
                return
        }
        w.Header().Set("Content-Type", "text/html; charset=utf-8")
        w.WriteHeader(http.StatusOK)
        _, _ = w.Write(content)
}

func (g *Gateway) handleBroadcastEvent(w http.ResponseWriter, request *http.Request) {
        var raw json.RawMessage
        if err := json.NewDecoder(request.Body).Decode(&raw); err != nil {
                writeJSON(w, http.StatusBadRequest, map[string]any{"error": "invalid json"})
                return
        }
        g.storeEvent(raw)
        writeJSON(w, http.StatusOK, map[string]any{"status": "broadcast"})
}

// storeEvent records one engine event under its run_id (so GET /api/tasks/:id
// returns the full history, not just the final line) and broadcasts it to
// every connected WebSocket client. Events without a run_id are broadcast only.
func (g *Gateway) storeEvent(raw json.RawMessage) {
        var probe struct {
                RunID string `json:"run_id"`
        }
        if json.Unmarshal(raw, &probe) == nil && probe.RunID != "" {
                g.taskEventsMu.Lock()
                events := g.taskEvents[probe.RunID]
                if len(events) < maxStoredEvents {
                        g.taskEvents[probe.RunID] = append(events, raw)
                }
                g.taskEventsMu.Unlock()
        }
        g.hub.Broadcast(raw)
}

func (g *Gateway) handleHealth(w http.ResponseWriter, _ *http.Request) {
        writeJSON(w, http.StatusOK, map[string]any{
                "status":  "ok",
                "service": "gateway",
                "clients": g.hub.Count(),
        })
}

// handleCreateTask validates the issue, stamps a run id, proxies to the
// orchestrator, and records the returned evidence pointer.
func (g *Gateway) handleCreateTask(w http.ResponseWriter, request *http.Request) {
        var body struct {
                Issue    string `json:"issue"`
                RepoRoot string `json:"repo_root"`
        }
        if err := json.NewDecoder(request.Body).Decode(&body); err != nil || body.Issue == "" {
                writeJSON(w, http.StatusBadRequest, map[string]any{
                        "error": "body must be JSON with a non-empty 'issue'",
                })
                return
        }
        runID := uuid.NewString()
        payload := map[string]any{"issue": body.Issue, "run_id": runID}
        if body.RepoRoot != "" {
                payload["repo_root"] = body.RepoRoot
        }
        proxied := proxyRequest(w, request, g.config.OrchestratorURL+"/agent/run", payload)
        if proxied == nil {
                writeJSON(w, http.StatusBadGateway, map[string]any{"error": "failed to communicate with orchestrator"})
                return
        }
        if errStr, hasErr := proxied["error"].(string); hasErr && errStr != "" && proxied["success"] == false {
                writeJSON(w, http.StatusInternalServerError, proxied)
                return
        }
        g.recordTask(runID, proxied)
        writeJSON(w, http.StatusCreated, map[string]any{
                "run_id": runID, "result": proxied,
        })
}

func (g *Gateway) handleTask(w http.ResponseWriter, request *http.Request) {
        runID := request.PathValue("id")
        g.taskEventsMu.Lock()
        events := g.taskEvents[runID]
        g.taskEventsMu.Unlock()
        if events == nil {
                writeJSON(w, http.StatusNotFound, map[string]any{
                        "error": "unknown run " + runID,
                })
                return
        }
        writeJSON(w, http.StatusOK, map[string]any{"run_id": runID, "events": events})
}

func (g *Gateway) handleMetrics(w http.ResponseWriter, _ *http.Request) {
        g.taskEventsMu.Lock()
        runs := len(g.taskEvents)
        g.taskEventsMu.Unlock()
        writeJSON(w, http.StatusOK, map[string]any{
                "runs": runs, "ws_clients": g.hub.Count(), "timestamp": time.Now().UTC(),
        })
}

// handleWS upgrades and registers a client; events arrive via Broadcast
// (fed by the Redis subscriber in main).
func (g *Gateway) handleWS(w http.ResponseWriter, request *http.Request) {
        upgrader := websocket.Upgrader{CheckOrigin: func(*http.Request) bool { return true }}
        conn, err := upgrader.Upgrade(w, request, nil)
        if err != nil {
                return
        }
        g.hub.add(conn)
        defer g.hub.remove(conn)
        for {
                if _, _, err := conn.ReadMessage(); err != nil {
                        return
                }
        }
}

// proxyToOrchestrator forwards any non-special-cased request to the
// Python orchestrator (TECHNICAL_IMPLEMENTATION.md §3.1 synchronous leg).
func (g *Gateway) proxyToOrchestrator(w http.ResponseWriter, request *http.Request) {
        target := g.config.OrchestratorURL + request.URL.Path
        if request.URL.RawQuery != "" {
                target += "?" + request.URL.RawQuery
        }
        proxyRequest(w, request, target, nil)
}

func proxyRequest(w http.ResponseWriter, request *http.Request, target string, body map[string]any) map[string]any {
        upstream, err := url.Parse(target)
        if err != nil {
                writeJSON(w, http.StatusBadGateway, map[string]any{"error": err.Error()})
                return nil
        }
        proxied := &httputil.ReverseProxy{Rewrite: func(pr *httputil.ProxyRequest) {
                pr.SetURL(upstream)
                pr.Out.Host = upstream.Host
                // SetURL JOINS the incoming path onto the target path (ReverseProxy
                // footgun): overwrite with the exact target path (live-run finding).
                pr.Out.URL.Path = upstream.Path
                pr.Out.URL.RawPath = ""
        }}
        if body == nil {
                // plain proxy: stream the upstream response straight through
                proxied.ServeHTTP(w, request)
                return nil
        }
        // capture mode: the caller wants the decoded body (and writes its own
        // response), so nothing is forwarded to the real writer here.
        encoded, _ := json.Marshal(body)
        proxied.Rewrite = func(pr *httputil.ProxyRequest) {
                pr.SetURL(upstream)
                pr.Out.Host = upstream.Host
                pr.Out.URL.Path = upstream.Path
                pr.Out.URL.RawPath = ""
                pr.Out.Method = http.MethodPost
                pr.Out.ContentLength = int64(len(encoded))
                pr.Out.Body = io.NopCloser(bytes.NewReader(encoded))
                pr.Out.Header.Set("Content-Type", "application/json")
        }
        recorder := httptest.NewRecorder()
        proxied.ServeHTTP(recorder, request)
        var decoded map[string]any
        if json.Unmarshal(recorder.Body.Bytes(), &decoded) == nil {
                return decoded
        }
        if recorder.Code >= 400 {
                return map[string]any{"error": string(recorder.Body.Bytes()), "success": false}
        }
        return nil
}

// recordTask appends the orchestrator's final response to the run's event
// history and broadcasts a task.completed event to WebSocket clients. The
// run_id threaded through /agent/run means the streamed events and this
// record share one map key.
func (g *Gateway) recordTask(runID string, result map[string]any) {
        encoded, _ := json.Marshal(map[string]any{
                "event": "task.completed", "run_id": runID, "result": result,
        })
        g.taskEventsMu.Lock()
        events := g.taskEvents[runID]
        if len(events) < maxStoredEvents {
                g.taskEvents[runID] = append(events, encoded)
        }
        g.taskEventsMu.Unlock()
        g.hub.Broadcast(encoded)
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
        w.Header().Set("Content-Type", "application/json")
        w.WriteHeader(status)
        _ = json.NewEncoder(w).Encode(payload)
}

func main() {
        config := LoadConfig()
        gateway := NewGateway(config, nil)
        go subscribeRedis(config, gateway)
        log.Printf("foreman gateway listening on %s (orchestrator: %s, redis: %s)",
                config.Addr, config.OrchestratorURL, config.RedisAddr)
        log.Fatal(http.ListenAndServe(config.Addr, gateway.Handler()))
}

// subscribeRedis feeds the WebSocket hub from the `harness.events.*` pub/sub
// bus (the documented platform event path). It is a no-op without REDIS_ADDR;
// on connect failure it retries with backoff while the orchestrator's direct
// HTTP POST fallback (events.py) keeps the cockpit live either way.
func subscribeRedis(config Config, gateway *Gateway) {
        if config.RedisAddr == "" {
                return
        }
        client := redis.NewClient(&redis.Options{Addr: config.RedisAddr})
        for {
                sub := client.PSubscribe(context.Background(), "harness.events.*")
                for msg := range sub.Channel() {
                        if msg == nil {
                                break
                        }
                        gateway.storeEvent(json.RawMessage(msg.Payload))
                }
                _ = sub.Close()
                log.Printf("gateway: redis subscription ended; reconnecting in 3s (%s)", config.RedisAddr)
                time.Sleep(3 * time.Second)
        }
}
