package main

import (
        "encoding/json"
        "net/http"
        "net/http/httptest"
        "strings"
        "testing"
)

func fakeOrchestrator(t *testing.T, status int, body map[string]any) *httptest.Server {
        t.Helper()
        return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
                w.Header().Set("Content-Type", "application/json")
                w.WriteHeader(status)
                _ = json.NewEncoder(w).Encode(body)
        }))
}

func TestHealthEndpoint(t *testing.T) {
        gateway := NewGateway(LoadConfig(), nil)
        server := httptest.NewServer(gateway.Handler())
        defer server.Close()

        response, err := http.Get(server.URL + "/api/health")
        if err != nil {
                t.Fatalf("health request failed: %v", err)
        }
        defer response.Body.Close()
        var body map[string]any
        if err := json.NewDecoder(response.Body).Decode(&body); err != nil {
                t.Fatalf("decode failed: %v", err)
        }
        if body["status"] != "ok" || body["service"] != "gateway" {
                t.Fatalf("unexpected health body: %v", body)
        }
}

func TestCreateTaskProxiesAndRecords(t *testing.T) {
        upstream := fakeOrchestrator(t, http.StatusOK, map[string]any{
                "success": true, "outcome": "VERIFIED", "evidence_path": "/tmp/evidence",
        })
        defer upstream.Close()

        gateway := NewGateway(Config{OrchestratorURL: upstream.URL}, nil)
        server := httptest.NewServer(gateway.Handler())
        defer server.Close()

        response, err := http.Post(server.URL+"/api/tasks", "application/json",
                strings.NewReader(`{"issue": "fix the parser", "repo_root": "/tmp/target"}`))
        if err != nil {
                t.Fatalf("create task failed: %v", err)
        }
        defer response.Body.Close()
        if response.StatusCode != http.StatusCreated {
                t.Fatalf("expected 201, got %d", response.StatusCode)
        }
        var body map[string]any
        _ = json.NewDecoder(response.Body).Decode(&body)
        runID, ok := body["run_id"].(string)
        if !ok || runID == "" {
                t.Fatalf("missing run_id in response: %v", body)
        }
        if body["result"].(map[string]any)["outcome"] != "VERIFIED" {
                t.Fatalf("proxied result missing: %v", body)
        }

        // the recorded task is retrievable
        taskResponse, err := http.Get(server.URL + "/api/tasks/" + runID)
        if err != nil {
                t.Fatalf("task lookup failed: %v", err)
        }
        defer taskResponse.Body.Close()
        if taskResponse.StatusCode != http.StatusOK {
                t.Fatalf("expected 200 for recorded task, got %d", taskResponse.StatusCode)
        }
}

func TestCreateTaskRejectsEmptyIssue(t *testing.T) {
        gateway := NewGateway(LoadConfig(), nil)
        server := httptest.NewServer(gateway.Handler())
        defer server.Close()

        response, err := http.Post(server.URL+"/api/tasks", "application/json",
                strings.NewReader(`{"issue": ""}`))
        if err != nil {
                t.Fatalf("request failed: %v", err)
        }
        defer response.Body.Close()
        if response.StatusCode != http.StatusBadRequest {
                t.Fatalf("expected 400 for empty issue, got %d", response.StatusCode)
        }
}

func TestUnknownTaskReturns404(t *testing.T) {
        gateway := NewGateway(LoadConfig(), nil)
        server := httptest.NewServer(gateway.Handler())
        defer server.Close()

        response, err := http.Get(server.URL + "/api/tasks/does-not-exist")
        if err != nil {
                t.Fatalf("request failed: %v", err)
        }
        defer response.Body.Close()
        if response.StatusCode != http.StatusNotFound {
                t.Fatalf("expected 404, got %d", response.StatusCode)
        }
}

func TestProxyForwardsToOrchestrator(t *testing.T) {
        upstream := fakeOrchestrator(t, http.StatusOK, map[string]any{
                "agents": []string{"architect-1", "ver-1"},
        })
        defer upstream.Close()

        gateway := NewGateway(Config{OrchestratorURL: upstream.URL}, nil)
        server := httptest.NewServer(gateway.Handler())
        defer server.Close()

        response, err := http.Get(server.URL + "/api/agents")
        if err != nil {
                t.Fatalf("proxy request failed: %v", err)
        }
        defer response.Body.Close()
        var body map[string]any
        _ = json.NewDecoder(response.Body).Decode(&body)
        if body["agents"] == nil {
                t.Fatalf("expected proxied agents payload: %v", body)
        }
}

func TestHubBroadcastReachesClients(t *testing.T) {
        hub := NewHub()
        if hub.Count() != 0 {
                t.Fatal("hub must start empty")
        }
        hub.Broadcast([]byte(`{"event": "no-clients-attached"}`)) // must not block
        if hub.Count() != 0 {
                t.Fatal("broadcast without clients must be a no-op")
        }
}

func TestDashboardEndpoint(t *testing.T) {
        gateway := NewGateway(LoadConfig(), nil)
        server := httptest.NewServer(gateway.Handler())
        defer server.Close()

        response, err := http.Get(server.URL + "/")
        if err != nil {
                t.Fatalf("dashboard request failed: %v", err)
        }
        defer response.Body.Close()
        if response.StatusCode != http.StatusOK {
                t.Fatalf("expected 200, got %d", response.StatusCode)
        }
}

func TestBroadcastEventEndpoint(t *testing.T) {
        gateway := NewGateway(LoadConfig(), nil)
        server := httptest.NewServer(gateway.Handler())
        defer server.Close()

        response, err := http.Post(server.URL+"/api/events", "application/json",
                strings.NewReader(`{"event": "test.event", "run_id": "r1"}`))
        if err != nil {
                t.Fatalf("broadcast request failed: %v", err)
        }
        defer response.Body.Close()
        if response.StatusCode != http.StatusOK {
                t.Fatalf("expected 200, got %d", response.StatusCode)
        }
}

func TestBroadcastEventStoresHistoryByRunID(t *testing.T) {
        // streamed events must be retrievable from GET /api/tasks/:id — the
        // cockpit's history panel and the API contract both depend on it.
        gateway := NewGateway(LoadConfig(), nil)
        server := httptest.NewServer(gateway.Handler())
        defer server.Close()

        for _, body := range []string{
                `{"event": "run.start", "run_id": "abc123"}`,
                `{"event": "architect.plan", "run_id": "abc123", "subtasks": ["st-1"]}`,
                `{"event": "run.end", "run_id": "abc123", "success": true}`,
                `{"event": "no-run-id-here"}`,
        } {
                resp, err := http.Post(server.URL+"/api/events", "application/json", strings.NewReader(body))
                if err != nil {
                        t.Fatalf("broadcast request failed: %v", err)
                }
                resp.Body.Close()
                if resp.StatusCode != http.StatusOK {
                        t.Fatalf("expected 200, got %d", resp.StatusCode)
                }
        }

        taskResponse, err := http.Get(server.URL + "/api/tasks/abc123")
        if err != nil {
                t.Fatalf("task lookup failed: %v", err)
        }
        defer taskResponse.Body.Close()
        if taskResponse.StatusCode != http.StatusOK {
                t.Fatalf("expected 200 for streamed run, got %d", taskResponse.StatusCode)
        }
        var body struct {
                RunID  string          `json:"run_id"`
                Events []map[string]any `json:"events"`
        }
        if err := json.NewDecoder(taskResponse.Body).Decode(&body); err != nil {
                t.Fatalf("decode failed: %v", err)
        }
        if body.RunID != "abc123" || len(body.Events) != 3 {
                t.Fatalf("expected 3 stored events for abc123, got %d (%v)", len(body.Events), body)
        }
}

func TestCreateTaskPassesRunIDThrough(t *testing.T) {
        // the gateway stamps a run id and the orchestrator must receive it, so
        // streamed events, evidence dir, and task record share ONE id.
        var received map[string]any
        upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
                if err := json.NewDecoder(r.Body).Decode(&received); err != nil {
                        t.Errorf("upstream decode failed: %v", err)
                }
                w.Header().Set("Content-Type", "application/json")
                _ = json.NewEncoder(w).Encode(map[string]any{
                        "run_id": received["run_id"], "success": true, "outcome": "VERIFIED",
                        "evidence_path": "/tmp/evidence",
                })
        }))
        defer upstream.Close()

        gateway := NewGateway(Config{OrchestratorURL: upstream.URL}, nil)
        server := httptest.NewServer(gateway.Handler())
        defer server.Close()

        response, err := http.Post(server.URL+"/api/tasks", "application/json",
                strings.NewReader(`{"issue": "fix the parser", "repo_root": "/tmp/target"}`))
        if err != nil {
                t.Fatalf("create task failed: %v", err)
        }
        defer response.Body.Close()
        var body map[string]any
        _ = json.NewDecoder(response.Body).Decode(&body)
        runID, _ := body["run_id"].(string)
        if runID == "" {
                t.Fatal("gateway must stamp a run id")
        }
        if received["run_id"] != runID {
                t.Fatalf("orchestrator received run id %v, gateway stamped %s", received["run_id"], runID)
        }
        // the task.completed record lands in the same history bucket
        taskResponse, err := http.Get(server.URL + "/api/tasks/" + runID)
        if err != nil {
                t.Fatalf("task lookup failed: %v", err)
        }
        defer taskResponse.Body.Close()
        var taskBody struct {
                Events []map[string]any `json:"events"`
        }
        if err := json.NewDecoder(taskResponse.Body).Decode(&taskBody); err != nil {
                t.Fatalf("task decode failed: %v", err)
        }
        if len(taskBody.Events) != 1 || taskBody.Events[0]["event"] != "task.completed" {
                t.Fatalf("expected one task.completed event, got %v", taskBody.Events)
        }
}

func TestEvidenceRouteProxiesToOrchestrator(t *testing.T) {
        upstream := fakeOrchestrator(t, http.StatusOK, map[string]any{
                "found": true, "run_id": "r9", "name": "patch.diff", "content": "+ fixed",
        })
        defer upstream.Close()

        gateway := NewGateway(Config{OrchestratorURL: upstream.URL}, nil)
        server := httptest.NewServer(gateway.Handler())
        defer server.Close()

        response, err := http.Get(server.URL + "/api/evidence/r9/file/patch.diff")
        if err != nil {
                t.Fatalf("evidence request failed: %v", err)
        }
        defer response.Body.Close()
        if response.StatusCode != http.StatusOK {
                t.Fatalf("expected 200, got %d", response.StatusCode)
        }
        var body map[string]any
        _ = json.NewDecoder(response.Body).Decode(&body)
        if body["found"] != true {
                t.Fatalf("expected proxied evidence payload, got %v", body)
        }
}

func TestStoreEventCapsHistory(t *testing.T) {
        gateway := NewGateway(LoadConfig(), nil)
        for i := 0; i < maxStoredEvents+50; i++ {
                raw, _ := json.Marshal(map[string]any{"event": "tick", "run_id": "capped"})
                gateway.storeEvent(raw)
        }
        gateway.taskEventsMu.Lock()
        count := len(gateway.taskEvents["capped"])
        gateway.taskEventsMu.Unlock()
        if count != maxStoredEvents {
                t.Fatalf("history must cap at %d events, got %d", maxStoredEvents, count)
        }
}
