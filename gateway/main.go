// Foreman API Gateway (platform P2, issue #71).
//
// TECHNICAL_IMPLEMENTATION.md §2.2 Service 1: HTTP routing, WebSocket
// broadcast of engine events, proxying to the Python orchestrator.
// Endpoints: GET /api/health, GET|POST /api/tasks, GET /api/tasks/:id,
// GET /api/agents, GET /api/metrics, WS /ws.
package main

import (
	"bufio"
	"bytes"
	"context"
	"embed"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"os"
	"sort"
	"strings"
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
	// EventsLogPath enables the append-only event journal: every stored
	// event is written through to disk and replayed on startup, so a
	// gateway restart no longer loses task history and the WS backlog.
	// Empty disables persistence (unit tests).
	EventsLogPath string
}

func LoadConfig() Config {
	return Config{
		Addr:            envOr("GATEWAY_ADDR", ":8080"),
		OrchestratorURL: envOr("ORCHESTRATOR_URL", "http://localhost:8000"),
		RedisAddr:       envOr("REDIS_ADDR", "localhost:6379"),
		EventsLogPath:   envOr("GATEWAY_EVENTS_LOG", "gateway-events.jsonl"),
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

const wsWriteTimeout = 5 * time.Second

func NewHub() *Hub {
	return &Hub{clients: make(map[*websocket.Conn]bool)}
}

func (h *Hub) remove(client *websocket.Conn) {
	h.mu.Lock()
	defer h.mu.Unlock()
	delete(h.clients, client)
}

// Broadcast writes one event to every client. A stuck or dead socket would
// otherwise block the whole hub (and, because storeEvent broadcasts while
// holding the event-history lock, event storage itself), so each write gets
// a deadline and failing clients are dropped instead of blocking everyone.
func (h *Hub) Broadcast(event []byte) {
	h.mu.Lock()
	defer h.mu.Unlock()
	for client := range h.clients {
		if err := writeEvent(client, event); err != nil {
			delete(h.clients, client)
		}
	}
}

// addWithBacklog registers a client and then replays the stored history of
// the latest run, so a cockpit connecting (or refreshing) mid-run or after a
// run rebuilds its full state: plan, tree, tokens, gates, verdict. The caller
// must hold g.taskEventsMu while calling this — the taskEventsMu -> hub.mu
// lock order (mirrored by storeEvent/recordTask) guarantees no live event is
// interleaved before, duplicated into, or lost between the snapshot and the
// registration.
func (h *Hub) addWithBacklog(client *websocket.Conn, backlog []json.RawMessage) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.clients[client] = true
	for _, event := range backlog {
		if writeEvent(client, event) != nil {
			delete(h.clients, client)
			return
		}
	}
}

func writeEvent(client *websocket.Conn, event []byte) error {
	_ = client.SetWriteDeadline(time.Now().Add(wsWriteTimeout))
	return client.WriteMessage(websocket.TextMessage, event)
}

func (h *Hub) Count() int {
	h.mu.Lock()
	defer h.mu.Unlock()
	return len(h.clients)
}

// Gateway wires the handlers to their dependencies (testable without network).
type Gateway struct {
	config       Config
	hub          *Hub
	upstream     *url.URL
	proxy        http.Handler
	taskEvents   map[string][]json.RawMessage // run_id -> events (memory + journal)
	taskEventsMu sync.Mutex
	latestRun    string            // most recently active run (WS replay source)
	runOrder     []string          // first-seen order of runs (journal order)
	runTitles    map[string]string // run_id -> custom title (survives replay)

	eventsWriter *bufio.Writer // journal writer (nil = disabled)
	eventsFile   *os.File
}

func NewGateway(config Config, proxy http.Handler) *Gateway {
	g := &Gateway{
		config:     config,
		hub:        NewHub(),
		proxy:      proxy,
		taskEvents: make(map[string][]json.RawMessage),
		runTitles:  make(map[string]string),
	}
	g.openEventJournal()
	g.loadRunTitles()
	return g
}

// maxStoredEvents caps one run's in-memory event history: the cockpit's
// feed is a live stream, the history is a convenience, and an unbounded map
// is a leak on a long-lived daemon. v2 cockpit events (agent.step/tool/usage
// per model call) push real runs into the hundreds—low thousands, so the
// cap sits above the largest observed real-run stream (~2k events at 4M
// tokens) while still bounding memory.
const maxStoredEvents = 2000

// Journal bounds: rotate at startup when the on-disk log exceeds
// maxJournalBytes (package var so tests can lower it), and replay only the
// last maxReplayedRuns runs so a long-lived journal does not translate into
// unbounded startup memory.
var maxJournalBytes int64 = 50 << 20 // 50 MiB

const maxReplayedRuns = 50

// openEventJournal attaches the append-only event journal: replay history
// from disk, then write through every stored event. Rotation (rename to
// .old) keeps the live journal bounded; a disabled journal (empty path or
// open failure) degrades to the previous in-memory-only behavior.
func (g *Gateway) openEventJournal() {
	if g.config.EventsLogPath == "" {
		return
	}
	if info, err := os.Stat(g.config.EventsLogPath); err == nil && info.Size() > maxJournalBytes {
		_ = os.Remove(g.config.EventsLogPath + ".old")
		if err := os.Rename(g.config.EventsLogPath, g.config.EventsLogPath+".old"); err != nil {
			log.Printf("gateway: event journal rotation failed (%v)", err)
		}
	}
	file, err := os.OpenFile(g.config.EventsLogPath, os.O_CREATE|os.O_RDWR|os.O_APPEND, 0o644)
	if err != nil {
		log.Printf("gateway: event journal disabled (%v)", err)
		return
	}
	g.replayJournal(file)
	g.eventsFile = file
	g.eventsWriter = bufio.NewWriter(file)
}

// replayJournal rebuilds taskEvents/latestRun from the journal file. Events
// without a run_id are skipped (they were broadcast-only in life). Per-run
// and total-run caps mirror the live in-memory bounds.
func (g *Gateway) replayJournal(file *os.File) {
	if _, err := file.Seek(0, io.SeekStart); err != nil {
		return
	}
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024) // run.start carries 2KB+ of issue text
	var order []string
	perRun := make(map[string][]json.RawMessage)
	for scanner.Scan() {
		line := bytes.TrimSpace(scanner.Bytes())
		if len(line) == 0 {
			continue
		}
		var probe struct {
			RunID string `json:"run_id"`
		}
		if json.Unmarshal(line, &probe) != nil || probe.RunID == "" {
			continue
		}
		if _, seen := perRun[probe.RunID]; !seen {
			order = append(order, probe.RunID)
		}
		if len(perRun[probe.RunID]) < maxStoredEvents {
			perRun[probe.RunID] = append(perRun[probe.RunID], json.RawMessage(append([]byte(nil), line...)))
		}
	}
	if len(order) > maxReplayedRuns {
		order = order[len(order)-maxReplayedRuns:]
	}
	for _, id := range order {
		g.taskEvents[id] = perRun[id]
	}
	g.runOrder = order
	if len(order) > 0 {
		g.latestRun = order[len(order)-1]
		log.Printf("gateway: replayed %d run(s) from event journal", len(order))
	}
}

// runTitlesPath is the custom-titles sidecar next to the event journal:
// renames must survive journal replay (they are session metadata, not
// engine events). Best-effort by design.
func (g *Gateway) runTitlesPath() string {
	if g.config.EventsLogPath == "" {
		return ""
	}
	return g.config.EventsLogPath + ".titles.json"
}

// loadRunTitles reads the sidecar; any damage yields empty titles.
func (g *Gateway) loadRunTitles() {
	path := g.runTitlesPath()
	if path == "" {
		return
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return
	}
	var titles map[string]string
	if json.Unmarshal(data, &titles) == nil {
		g.runTitles = titles
	}
}

// saveRunTitlesLocked persists the sidecar (caller holds taskEventsMu).
func (g *Gateway) saveRunTitlesLocked() {
	path := g.runTitlesPath()
	if path == "" {
		return
	}
	encoded, _ := json.Marshal(g.runTitles)
	if err := os.WriteFile(path, encoded, 0o644); err != nil {
		log.Printf("gateway: run-titles sidecar write failed (%v)", err)
	}
}

// compactJournalLocked rewrites the journal WITHOUT one run (chat delete).
// The writer is re-anchored on the fresh file; a missing/disabled journal is
// a no-op. Caller holds taskEventsMu.
func (g *Gateway) compactJournalLocked(dropRunID string) {
	if g.config.EventsLogPath == "" || g.eventsWriter == nil {
		return
	}
	path := g.config.EventsLogPath
	_ = g.eventsWriter.Flush()
	_ = g.eventsFile.Close()
	g.eventsWriter = nil
	g.eventsFile = nil
	data, err := os.ReadFile(path)
	if err != nil {
		g.disableJournal("compact read: " + err.Error())
		return
	}
	var kept []byte
	for _, line := range bytes.Split(data, []byte{'\n'}) {
		if len(bytes.TrimSpace(line)) == 0 {
			continue
		}
		var probe struct {
			RunID string `json:"run_id"`
		}
		if json.Unmarshal(line, &probe) == nil && probe.RunID == dropRunID {
			continue
		}
		kept = append(kept, line...)
		kept = append(kept, '\n')
	}
	if err := os.WriteFile(path, kept, 0o644); err != nil {
		g.disableJournal("compact write: " + err.Error())
		return
	}
	file, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR|os.O_APPEND, 0o644)
	if err != nil {
		g.disableJournal("compact reopen: " + err.Error())
		return
	}
	g.eventsFile = file
	g.eventsWriter = bufio.NewWriter(file)
}

// journalEvent writes one stored event through to disk (called under
// taskEventsMu, same critical section as the in-memory append). The first
// write failure disables the journal for the process lifetime: persistence
// must never take the live cockpit down with it.
func (g *Gateway) journalEvent(raw json.RawMessage) {
	if g.eventsWriter == nil {
		return
	}
	if _, err := g.eventsWriter.Write(append(raw, '\n')); err != nil {
		g.disableJournal(fmt.Sprintf("write: %v", err))
		return
	}
	if err := g.eventsWriter.Flush(); err != nil {
		g.disableJournal(fmt.Sprintf("flush: %v", err))
	}
}

func (g *Gateway) disableJournal(reason string) {
	log.Printf("gateway: event journal disabled: %s", reason)
	g.eventsWriter = nil
	if g.eventsFile != nil {
		_ = g.eventsFile.Close()
		g.eventsFile = nil
	}
}

// CloseEventJournal flushes and closes the journal (tests and graceful
// shutdown; a no-op when disabled).
func (g *Gateway) CloseEventJournal() {
	g.taskEventsMu.Lock()
	defer g.taskEventsMu.Unlock()
	if g.eventsWriter != nil {
		_ = g.eventsWriter.Flush()
	}
	g.eventsWriter = nil
	if g.eventsFile != nil {
		_ = g.eventsFile.Close()
		g.eventsFile = nil
	}
}

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
	mux.HandleFunc("GET /api/models", g.proxyToOrchestrator)
	mux.HandleFunc("POST /api/projects", g.proxyToOrchestrator)
	mux.HandleFunc("GET /api/projects", g.proxyToOrchestrator)
	mux.HandleFunc("GET /api/projects/{rest...}", g.proxyToOrchestrator)
	mux.HandleFunc("DELETE /api/projects/{rest...}", g.proxyToOrchestrator)
	mux.HandleFunc("GET /api/evidence/{rest...}", g.proxyToOrchestrator)
	mux.HandleFunc("GET /api/runs", g.handleRunsList)
	mux.HandleFunc("PATCH /api/runs/{id}", g.handleRunRename)
	mux.HandleFunc("DELETE /api/runs/{id}", g.handleRunDelete)
	mux.HandleFunc("POST /api/runs/{id}/cancel", g.handleRunCancel)
	mux.HandleFunc("GET /api/metrics", g.handleMetrics)
	mux.HandleFunc("GET /ws", g.handleWS)
	// Replay fixtures for both cockpits' offline demo mode
	// (?replay=events.sample.jsonl). Served from the same embedded FS
	// as the dashboard so a static dev server is never required.
	mux.HandleFunc("GET /events/{name}", g.handleEventFixture)
	return mux
}

// handleEventFixture serves one replay fixture (.jsonl) from the embedded
// web/ directory. Only .jsonl basenames: no directories, no traversal, and
// the dashboard itself (index.html) is not reachable through this route.
func (g *Gateway) handleEventFixture(w http.ResponseWriter, request *http.Request) {
	name := request.PathValue("name")
	if name == "" || !strings.HasSuffix(name, ".jsonl") || strings.ContainsAny(name, `/\`) {
		http.NotFound(w, request)
		return
	}
	content, err := webFS.ReadFile("web/" + name)
	if err != nil {
		http.NotFound(w, request)
		return
	}
	w.Header().Set("Content-Type", "application/x-ndjson")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(content)
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
// The broadcast happens WHILE holding taskEventsMu: handleWS snapshots the
// backlog and registers the client under the same lock (taskEventsMu -> hub.mu
// order everywhere), which is what makes the WS replay gapless — no event can
// slip between a client's snapshot and its registration.
func (g *Gateway) storeEvent(raw json.RawMessage) {
	var probe struct {
		RunID string `json:"run_id"`
	}
	g.taskEventsMu.Lock()
	defer g.taskEventsMu.Unlock()
	if json.Unmarshal(raw, &probe) == nil && probe.RunID != "" {
		events := g.taskEvents[probe.RunID]
		if len(events) < maxStoredEvents {
			g.taskEvents[probe.RunID] = append(events, raw)
		}
		if _, seen := g.seenRunLocked(probe.RunID); !seen {
			g.runOrder = append(g.runOrder, probe.RunID)
		}
		g.latestRun = probe.RunID
	}
	g.journalEvent(raw)
	g.hub.Broadcast(raw)
}

// seenRunLocked reports whether a run is tracked in runOrder.
func (g *Gateway) seenRunLocked(runID string) (int, bool) {
	for index, id := range g.runOrder {
		if id == runID {
			return index, true
		}
	}
	return -1, false
}

func (g *Gateway) handleHealth(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"status":  "ok",
		"service": "gateway",
		"clients": g.hub.Count(),
	})
}

// handleCreateTask validates the issue, stamps a run id, answers 202
// immediately, and runs the orchestrator's /agent/run in the background.
// Real-model runs take minutes: a synchronous response meant a browser fetch
// hanging for the whole run (and lying "launch failed" the moment any
// intermediary timed it out), plus a client disconnect could sever the
// upstream run mid-flight. The final orchestrator response is recorded and
// broadcast as a task.completed event when it arrives.
func (g *Gateway) handleCreateTask(w http.ResponseWriter, request *http.Request) {
	var body struct {
		Issue    string `json:"issue"`
		RepoRoot string `json:"repo_root"`
		// demo_mode opts the run into scripted model responses: the
		// cockpit's offline demo path (docs/eval-runbook §3). Forwarded
		// verbatim to the orchestrator's /agent/run contract.
		DemoMode bool `json:"demo_mode"`
		// Session management (platform P4): the run's display title
		// (cockpit chat list), the model profile powering it, and a
		// prior run id for chat-style continuation.
		Title        string `json:"title"`
		ModelProfile string `json:"model_profile"`
		FollowupOf   string `json:"followup_of"`
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
	if body.DemoMode {
		payload["demo_mode"] = true
	}
	if body.ModelProfile != "" {
		payload["model_profile"] = body.ModelProfile
	}
	if body.FollowupOf != "" {
		payload["followup_of"] = body.FollowupOf
	}
	// run.meta: the session-list record, journaled + broadcast BEFORE
	// the run starts, so the chat sidebar shows the session immediately
	// and GET /api/runs can rebuild after a restart. "title" is the
	// display name; the issue preview rides along for default naming.
	title := body.Title
	if title == "" {
		title = issuePreview(body.Issue, 70)
	}
	g.storeEvent(mustJSON(map[string]any{
		"event":         "run.meta",
		"run_id":        runID,
		"title":         title,
		"issue_preview": issuePreview(body.Issue, 140),
		"repo_root":     body.RepoRoot,
		"model_profile": body.ModelProfile,
		"followup_of":   body.FollowupOf,
		"demo":          body.DemoMode,
		"created_at":    time.Now().UTC().Format(time.RFC3339),
		"status":        "running",
	}))
	g.broadcastRunsChanged()
	writeJSON(w, http.StatusAccepted, map[string]any{
		"run_id": runID, "status": "accepted", "title": title,
		"detail": "run started; stream /ws and watch for task.completed",
	})
	// The client's request context dies with the response: the upstream
	// run must outlive it, so proxy from a detached context.
	go func() {
		bg := request.Clone(context.Background())
		bg.Body = http.NoBody
		proxied := g.captureUpstream(bg, g.config.OrchestratorURL+"/agent/run", payload)
		if proxied == nil {
			proxied = map[string]any{
				"error":   "failed to communicate with orchestrator",
				"success": false,
			}
		}
		g.recordTask(runID, proxied)
		g.broadcastRunsChanged()
	}()
}

func (g *Gateway) handleTask(w http.ResponseWriter, request *http.Request) {
	runID := request.PathValue("id")
	g.taskEventsMu.Lock()
	events := g.taskEvents[runID]
	if _, seen := g.seenRunLocked(runID); !seen && events == nil {
		g.taskEventsMu.Unlock()
		writeJSON(w, http.StatusNotFound, map[string]any{
			"error": "unknown run " + runID,
		})
		return
	}
	g.taskEventsMu.Unlock()
	writeJSON(w, http.StatusOK, map[string]any{"run_id": runID, "events": events})
}

// runSummary is one session in the chat list.
type runSummary struct {
	RunID        string `json:"run_id"`
	Title        string `json:"title"`
	IssuePreview string `json:"issue_preview"`
	RepoRoot     string `json:"repo_root"`
	ModelProfile string `json:"model_profile"`
	FollowupOf   string `json:"followup_of"`
	Status       string `json:"status"`
	Outcome      string `json:"outcome"`
	EventCount   int    `json:"event_count"`
	CreatedAt    string `json:"created_at"`
	UpdatedAt    string `json:"updated_at"`
}

// handleRunsList serves the chat session list (GET /api/runs): every run in
// memory+journal, newest last-active first, with custom titles applied.
func (g *Gateway) handleRunsList(w http.ResponseWriter, _ *http.Request) {
	g.taskEventsMu.Lock()
	summaries := make([]runSummary, 0, len(g.taskEvents))
	for runID, events := range g.taskEvents {
		summaries = append(summaries, g.summarizeRunLocked(runID, events))
	}
	g.taskEventsMu.Unlock()
	sort.Slice(summaries, func(i, j int) bool {
		return summaries[i].UpdatedAt > summaries[j].UpdatedAt
	})
	writeJSON(w, http.StatusOK, map[string]any{"runs": summaries})
}

// summarizeRunLocked derives one session row from the run's stored events.
// The caller holds taskEventsMu.
func (g *Gateway) summarizeRunLocked(runID string, events []json.RawMessage) runSummary {
	summary := runSummary{RunID: runID, EventCount: len(events), Status: "interrupted"}
	for _, raw := range events {
		var probe struct {
			Event        string         `json:"event"`
			Title        string         `json:"title"`
			IssuePreview string         `json:"issue_preview"`
			Issue        string         `json:"issue"`
			RepoRoot     string         `json:"repo_root"`
			ModelProfile string         `json:"model_profile"`
			FollowupOf   string         `json:"followup_of"`
			CreatedAt    string         `json:"created_at"`
			Success      bool           `json:"success"`
			Result       map[string]any `json:"result"`
		}
		if json.Unmarshal(raw, &probe) != nil {
			continue
		}
		switch probe.Event {
		case "run.meta":
			summary.Title = probe.Title
			summary.IssuePreview = probe.IssuePreview
			summary.RepoRoot = probe.RepoRoot
			summary.ModelProfile = probe.ModelProfile
			summary.FollowupOf = probe.FollowupOf
			summary.CreatedAt = probe.CreatedAt
		case "run.start":
			if summary.IssuePreview == "" {
				summary.IssuePreview = issuePreview(probe.Issue, 140)
			}
			if summary.RepoRoot == "" {
				summary.RepoRoot = probe.RepoRoot
			}
			if summary.CreatedAt == "" {
				summary.CreatedAt = time.Now().UTC().Format(time.RFC3339)
			}
			summary.Status = "running"
		case "run.end":
			summary.Status = "verified"
			if !probe.Success {
				summary.Status = "failed"
			}
		case "task.completed":
			if probe.Result != nil {
				if success, ok := probe.Result["success"].(bool); ok {
					if success {
						summary.Status = "verified"
					} else {
						summary.Status = "failed"
					}
				}
				if outcome, ok := probe.Result["outcome"].(string); ok {
					summary.Outcome = outcome
				}
			}
		case "run.failed":
			if summary.Status == "" || summary.Status == "running" {
				summary.Status = "failed"
			}
		}
		summary.UpdatedAt = time.Now().UTC().Format(time.RFC3339)
	}
	if g.runTitles != nil {
		if custom, ok := g.runTitles[runID]; ok {
			summary.Title = custom
		}
	}
	if summary.Title == "" {
		summary.Title = issuePreview(summary.IssuePreview, 70)
	}
	return summary
}

// handleRunRename renames a session (PATCH /api/runs/{id} {"title": ...}).
// Titles persist in a small sidecar so renames survive journal replay.
func (g *Gateway) handleRunRename(w http.ResponseWriter, request *http.Request) {
	runID := request.PathValue("id")
	var body struct {
		Title string `json:"title"`
	}
	if err := json.NewDecoder(request.Body).Decode(&body); err != nil || body.Title == "" {
		writeJSON(w, http.StatusBadRequest, map[string]any{
			"error": "body must be JSON with a non-empty 'title'",
		})
		return
	}
	g.taskEventsMu.Lock()
	if _, known := g.taskEvents[runID]; !known {
		g.taskEventsMu.Unlock()
		writeJSON(w, http.StatusNotFound, map[string]any{"error": "unknown run " + runID})
		return
	}
	if g.runTitles == nil {
		g.runTitles = map[string]string{}
	}
	g.runTitles[runID] = body.Title
	g.saveRunTitlesLocked()
	g.taskEventsMu.Unlock()
	g.broadcastRunsChanged()
	writeJSON(w, http.StatusOK, map[string]any{"run_id": runID, "title": body.Title})
}

// handleRunDelete removes a session from memory AND from the journal
// (rewritten without that run): the chat list's delete button.
func (g *Gateway) handleRunDelete(w http.ResponseWriter, request *http.Request) {
	runID := request.PathValue("id")
	g.taskEventsMu.Lock()
	if _, known := g.taskEvents[runID]; !known {
		g.taskEventsMu.Unlock()
		writeJSON(w, http.StatusNotFound, map[string]any{"error": "unknown run " + runID})
		return
	}
	delete(g.taskEvents, runID)
	delete(g.runTitles, runID)
	g.saveRunTitlesLocked()
	if g.latestRun == runID {
		g.latestRun = g.latestRemainingRunLocked()
	}
	g.compactJournalLocked(runID)
	g.taskEventsMu.Unlock()
	g.broadcastRunsChanged()
	writeJSON(w, http.StatusOK, map[string]any{"run_id": runID, "deleted": true})
}

// handleRunCancel proxies the chat stop button to the orchestrator's
// /agent/cancel; the run's own events (run.failed/run.end) arrive through
// the normal event path.
func (g *Gateway) handleRunCancel(w http.ResponseWriter, request *http.Request) {
	runID := request.PathValue("id")
	bg := request.Clone(context.Background())
	bg.Body = http.NoBody
	result := g.captureUpstream(
		bg, g.config.OrchestratorURL+"/agent/cancel", map[string]any{"run_id": runID},
	)
	if result == nil {
		writeJSON(w, http.StatusBadGateway, map[string]any{
			"error": "cannot reach orchestrator", "cancelled": false,
		})
		return
	}
	writeJSON(w, http.StatusOK, result)
}

// latestRemainingRunLocked picks the newest still-tracked run after a
// delete (caller holds taskEventsMu; runOrder preserves journal order).
func (g *Gateway) latestRemainingRunLocked() string {
	for i := len(g.runOrder) - 1; i >= 0; i-- {
		if _, alive := g.taskEvents[g.runOrder[i]]; alive {
			return g.runOrder[i]
		}
	}
	return ""
}

// broadcastRunsChanged nudges every cockpit to refetch the session list.
func (g *Gateway) broadcastRunsChanged() {
	g.storeEvent(mustJSON(map[string]any{
		"event": "runs.changed", "detail": "session list changed; refetch /api/runs",
	}))
}

// mustJSON encodes without failure paths (map[string]any is always valid).
func mustJSON(payload map[string]any) []byte {
	encoded, _ := json.Marshal(payload)
	return encoded
}

// issuePreview condenses issue text into a single-line title.
func issuePreview(issue string, limit int) string {
	compact := strings.Join(strings.Fields(issue), " ")
	if len(compact) > limit {
		return compact[:limit-1] + "…"
	}
	return compact
}

func (g *Gateway) handleMetrics(w http.ResponseWriter, _ *http.Request) {
	g.taskEventsMu.Lock()
	runs := len(g.taskEvents)
	g.taskEventsMu.Unlock()
	writeJSON(w, http.StatusOK, map[string]any{
		"runs": runs, "ws_clients": g.hub.Count(), "timestamp": time.Now().UTC(),
	})
}

// handleWS upgrades and registers a client; live events arrive via Broadcast
// (fed by POST /api/events and the Redis subscriber in main). On connect the
// client first receives the stored backlog of the latest run — a cockpit
// opened (or refreshed) mid-run or after a run therefore renders the FULL
// picture: roster, plan, tree, tokens, gates, verdict. run.start resets
// cockpit state, so replaying a whole run is idempotent for both cockpits.
// ?run_id=<id> selects a DIFFERENT session's backlog (chat list switching).
func (g *Gateway) handleWS(w http.ResponseWriter, request *http.Request) {
	upgrader := websocket.Upgrader{CheckOrigin: func(*http.Request) bool { return true }}
	conn, err := upgrader.Upgrade(w, request, nil)
	if err != nil {
		return
	}
	defer g.hub.remove(conn)
	selected := request.URL.Query().Get("run_id")
	g.taskEventsMu.Lock()
	var backlog []json.RawMessage
	source := g.latestRun
	if selected != "" {
		source = selected
	}
	if source != "" {
		backlog = append(backlog, g.taskEvents[source]...)
	}
	g.hub.addWithBacklog(conn, backlog) // hub.mu taken inside; taskEventsMu -> hub.mu order
	g.taskEventsMu.Unlock()
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
	proxyRequest(w, request, target)
}

// proxyRequest streams any non-special-cased request straight through to
// the orchestrator (GET /api/agents, evidence files). Capture-style calls
// with a request body use captureUpstream instead.
func proxyRequest(w http.ResponseWriter, request *http.Request, target string) {
	upstream, err := url.Parse(target)
	if err != nil {
		writeJSON(w, http.StatusBadGateway, map[string]any{"error": err.Error()})
		return
	}
	proxied := &httputil.ReverseProxy{Rewrite: func(pr *httputil.ProxyRequest) {
		pr.SetURL(upstream)
		pr.Out.Host = upstream.Host
		// SetURL JOINS the incoming path onto the target path (ReverseProxy
		// footgun): overwrite with the exact target path (live-run finding).
		pr.Out.URL.Path = upstream.Path
		pr.Out.URL.RawPath = ""
	}}
	proxied.ServeHTTP(w, request)
}

// captureUpstream POSTs body to target and returns the decoded JSON response
// (nil on transport failure or an unparseable body). It never writes to a
// client ResponseWriter — async callers have already answered the client by
// the time the upstream responds. base only contributes headers; its body is
// replaced and its context must be the caller's (detached for background runs).
func (g *Gateway) captureUpstream(base *http.Request, target string, body map[string]any) map[string]any {
	upstream, err := url.Parse(target)
	if err != nil {
		return map[string]any{"error": err.Error(), "success": false}
	}
	encoded, _ := json.Marshal(body)
	proxied := &httputil.ReverseProxy{Rewrite: func(pr *httputil.ProxyRequest) {
		pr.SetURL(upstream)
		pr.Out.Host = upstream.Host
		pr.Out.URL.Path = upstream.Path
		pr.Out.URL.RawPath = ""
		pr.Out.Method = http.MethodPost
		pr.Out.ContentLength = int64(len(encoded))
		pr.Out.Body = io.NopCloser(bytes.NewReader(encoded))
		pr.Out.Header.Set("Content-Type", "application/json")
	}}
	recorder := httptest.NewRecorder()
	proxied.ServeHTTP(recorder, base)
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
// record share one map key. The broadcast happens while holding taskEventsMu
// (same taskEventsMu -> hub.mu lock order as storeEvent) so a client
// connecting between the run's last event and this record still receives
// both, in order, via the backlog replay.
func (g *Gateway) recordTask(runID string, result map[string]any) {
	encoded, _ := json.Marshal(map[string]any{
		"event": "task.completed", "run_id": runID, "result": result,
	})
	g.taskEventsMu.Lock()
	defer g.taskEventsMu.Unlock()
	events := g.taskEvents[runID]
	if len(events) < maxStoredEvents {
		g.taskEvents[runID] = append(events, encoded)
	}
	g.latestRun = runID
	g.journalEvent(encoded)
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
