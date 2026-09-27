package main

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/websocket"
)

// waitFor retries cond until it holds or the deadline passes — task creation
// is asynchronous (202 + background proxy), so tests poll for the orchestrator
// interaction instead of asserting it synchronously.
func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", what)
}

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

func TestCreateTaskIsAsyncAndRecords(t *testing.T) {
	// POST /api/tasks answers 202 + run_id immediately (real runs take
	// minutes); the orchestrator result lands later as a recorded
	// task.completed event retrievable from GET /api/tasks/:id.
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
	if response.StatusCode != http.StatusAccepted {
		t.Fatalf("expected 202, got %d", response.StatusCode)
	}
	var body map[string]any
	_ = json.NewDecoder(response.Body).Decode(&body)
	runID, ok := body["run_id"].(string)
	if !ok || runID == "" {
		t.Fatalf("missing run_id in response: %v", body)
	}

	// the recorded task is retrievable once the background proxy completes
	var events []map[string]any
	waitFor(t, "task.completed to be recorded", func() bool {
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
		events = taskBody.Events
		return len(events) == 1
	})
	completed, _ := events[0]["result"].(map[string]any)
	if completed["outcome"] != "VERIFIED" {
		t.Fatalf("proxied result missing: %v", events)
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

func TestCreateTaskForwardsDemoMode(t *testing.T) {
	// The cockpit's demo checkbox only works if the gateway forwards the
	// flag instead of silently dropping it (integration finding: demo runs
	// were hitting the real model because demo_mode never crossed the proxy).
	received := make(chan map[string]any, 1)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		received <- body
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"success": true, "outcome": "DEMO"})
	}))
	defer upstream.Close()

	gateway := NewGateway(Config{OrchestratorURL: upstream.URL}, nil)
	server := httptest.NewServer(gateway.Handler())
	defer server.Close()

	response, err := http.Post(server.URL+"/api/tasks", "application/json",
		strings.NewReader(`{"issue": "demo run", "repo_root": "/tmp/r", "demo_mode": true}`))
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusAccepted {
		t.Fatalf("expected 202, got %d", response.StatusCode)
	}
	select {
	case body := <-received:
		if body["demo_mode"] != true {
			t.Fatalf("demo_mode must be forwarded, got %v", body["demo_mode"])
		}
		if body["run_id"] == "" || body["run_id"] == nil {
			t.Fatalf("run_id must be stamped, got %v", body["run_id"])
		}
	case <-time.After(3 * time.Second):
		t.Fatal("orchestrator never received the request")
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
		RunID  string           `json:"run_id"`
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
	received := make(chan map[string]any, 1)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("upstream decode failed: %v", err)
		}
		received <- body
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"run_id": body["run_id"], "success": true, "outcome": "VERIFIED",
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
	select {
	case upstreamBody := <-received:
		if upstreamBody["run_id"] != runID {
			t.Fatalf("orchestrator received run id %v, gateway stamped %s", upstreamBody["run_id"], runID)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("orchestrator never received the request")
	}
	// the task.completed record lands in the same history bucket
	var events []map[string]any
	waitFor(t, "task.completed to be recorded", func() bool {
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
		events = taskBody.Events
		return len(events) == 1
	})
	if events[0]["event"] != "task.completed" {
		t.Fatalf("expected one task.completed event, got %v", events)
	}
}

func TestWSConnectReplaysBacklog(t *testing.T) {
	// A cockpit connecting mid-run (or refreshing after a run) must receive
	// the latest run's stored history on connect — otherwise the page starts
	// blank and can never render the plan, tree, tokens, or verdict. Live
	// events must continue seamlessly after the replay.
	gateway := NewGateway(LoadConfig(), nil)
	server := httptest.NewServer(gateway.Handler())
	defer server.Close()

	for _, body := range []string{
		`{"event": "run.start", "run_id": "back1"}`,
		`{"event": "architect.plan", "run_id": "back1", "subtasks": ["st-1"]}`,
	} {
		resp, err := http.Post(server.URL+"/api/events", "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatalf("broadcast request failed: %v", err)
		}
		resp.Body.Close()
	}

	wsURL := "ws" + strings.TrimPrefix(server.URL, "http") + "/ws"
	conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
	if err != nil {
		t.Fatalf("ws dial failed: %v", err)
	}
	defer conn.Close()
	_ = conn.SetReadDeadline(time.Now().Add(3 * time.Second))

	for i, want := range []string{"run.start", "architect.plan"} {
		_, msg, err := conn.ReadMessage()
		if err != nil {
			t.Fatalf("replayed event %d missing: %v", i, err)
		}
		var ev map[string]any
		if err := json.Unmarshal(msg, &ev); err != nil {
			t.Fatalf("replayed event %d not json: %v", i, err)
		}
		if ev["event"] != want || ev["run_id"] != "back1" {
			t.Fatalf("expected replayed %s for back1, got %v", want, ev)
		}
	}

	// live events still flow after the replay
	resp, err := http.Post(server.URL+"/api/events", "application/json",
		strings.NewReader(`{"event": "run.end", "run_id": "back1", "success": true}`))
	if err != nil {
		t.Fatalf("broadcast request failed: %v", err)
	}
	resp.Body.Close()
	_, msg, err := conn.ReadMessage()
	if err != nil {
		t.Fatalf("live event after replay missing: %v", err)
	}
	var live map[string]any
	_ = json.Unmarshal(msg, &live)
	if live["event"] != "run.end" {
		t.Fatalf("expected live run.end, got %v", live)
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

func TestEventFixtureRoute(t *testing.T) {
	gateway := NewGateway(LoadConfig(), nil)
	server := httptest.NewServer(gateway.Handler())
	defer server.Close()

	response, err := http.Get(server.URL + "/events/events.sample.jsonl")
	if err != nil {
		t.Fatalf("fixture fetch failed: %v", err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		t.Fatalf("expected 200 for embedded fixture, got %d", response.StatusCode)
	}
	body, _ := io.ReadAll(response.Body)
	if !strings.Contains(string(body), "run.start") {
		t.Fatal("fixture content missing run.start line")
	}

	// non-jsonl and traversal-ish names are rejected
	for _, bad := range []string{"/events/index.html", "/events/../../main.go", "/events/nope.jsonl"} {
		resp, err := http.Get(server.URL + bad)
		if err != nil {
			t.Fatalf("request %s failed: %v", bad, err)
		}
		resp.Body.Close()
		if resp.StatusCode != http.StatusNotFound {
			t.Fatalf("expected 404 for %s, got %d", bad, resp.StatusCode)
		}
	}
}
