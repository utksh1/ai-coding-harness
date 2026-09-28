package main

// Session-management API tests (platform P4): the chat list, renames that
// survive restarts, deletes that compact the journal, cancellation proxying,
// and WS backlog selection by run_id.

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/websocket"
)

func TestRunsListSummarizesSessions(t *testing.T) {
	gateway := NewGateway(testConfig(), nil)
	gateway.storeEvent(mustJSON(map[string]any{
		"event": "run.meta", "run_id": "r1", "title": "fix the parser",
		"issue_preview": "fix the parser", "repo_root": "/tmp/target",
		"created_at": "2026-01-01T00:00:00Z", "status": "running",
	}))
	gateway.storeEvent(mustJSON(map[string]any{
		"event": "run.end", "run_id": "r1", "success": true,
	}))
	gateway.storeEvent(mustJSON(map[string]any{
		"event": "run.meta", "run_id": "r2", "title": "second task",
		"issue_preview": "second task", "created_at": "2026-01-02T00:00:00Z",
	}))

	server := httptest.NewServer(gateway.Handler())
	defer server.Close()

	response, err := http.Get(server.URL + "/api/runs")
	if err != nil {
		t.Fatalf("runs list failed: %v", err)
	}
	defer response.Body.Close()
	var body struct {
		Runs []runSummary `json:"runs"`
	}
	if err := json.NewDecoder(response.Body).Decode(&body); err != nil {
		t.Fatalf("decode failed: %v", err)
	}
	if len(body.Runs) != 2 {
		t.Fatalf("expected 2 runs, got %d", len(body.Runs))
	}
	byID := map[string]runSummary{}
	for _, run := range body.Runs {
		byID[run.RunID] = run
	}
	if byID["r1"].Status != "verified" {
		t.Fatalf("r1 should be verified, got %s", byID["r1"].Status)
	}
	if byID["r1"].Title != "fix the parser" {
		t.Fatalf("r1 title wrong: %s", byID["r1"].Title)
	}
	if byID["r2"].Status != "interrupted" {
		t.Fatalf("r2 (meta only, no terminal event) should be interrupted, got %s", byID["r2"].Status)
	}
}

func TestRunRenamePersistsAcrossRestart(t *testing.T) {
	dir := t.TempDir()
	journal := filepath.Join(dir, "events.jsonl")
	cfg := testConfig()
	cfg.EventsLogPath = journal

	gateway := NewGateway(cfg, nil)
	gateway.storeEvent(mustJSON(map[string]any{
		"event": "run.meta", "run_id": "r1", "title": "original", "issue_preview": "original",
	}))
	server := httptest.NewServer(gateway.Handler())
	defer server.Close()

	patch, err := http.NewRequest("PATCH", server.URL+"/api/runs/r1",
		strings.NewReader(`{"title": "renamed session"}`))
	if err != nil {
		t.Fatalf("patch build failed: %v", err)
	}
	response, err := http.DefaultClient.Do(patch)
	if err != nil || response.StatusCode != http.StatusOK {
		t.Fatalf("rename failed: %v %v", err, response)
	}
	response.Body.Close()
	gateway.CloseEventJournal()

	// Restart: titles replay from the sidecar, not the journal.
	restarted := NewGateway(cfg, nil)
	defer restarted.CloseEventJournal()
	restored := restarted.runTitles["r1"]
	if restored != "renamed session" {
		t.Fatalf("rename did not survive restart: %q", restored)
	}
}

func TestRunDeleteCompactsJournalAndMemory(t *testing.T) {
	dir := t.TempDir()
	journal := filepath.Join(dir, "events.jsonl")
	cfg := testConfig()
	cfg.EventsLogPath = journal

	gateway := NewGateway(cfg, nil)
	defer gateway.CloseEventJournal()
	gateway.storeEvent(mustJSON(map[string]any{"event": "run.meta", "run_id": "keep", "title": "keep"}))
	gateway.storeEvent(mustJSON(map[string]any{"event": "run.meta", "run_id": "drop", "title": "drop"}))
	gateway.storeEvent(mustJSON(map[string]any{"event": "run.end", "run_id": "drop", "success": true}))

	server := httptest.NewServer(gateway.Handler())
	defer server.Close()

	request, _ := http.NewRequest("DELETE", server.URL+"/api/runs/drop", nil)
	response, err := http.DefaultClient.Do(request)
	if err != nil || response.StatusCode != http.StatusOK {
		t.Fatalf("delete failed: %v %v", err, response)
	}
	response.Body.Close()

	gateway.taskEventsMu.Lock()
	_, dropAlive := gateway.taskEvents["drop"]
	_, keepAlive := gateway.taskEvents["keep"]
	gateway.taskEventsMu.Unlock()
	if dropAlive {
		t.Fatal("deleted run still in memory")
	}
	if !keepAlive {
		t.Fatal("surviving run was collateral damage")
	}
	if gateway.latestRun != "keep" {
		t.Fatalf("latestRun should fall back to keep, got %q", gateway.latestRun)
	}

	// Restart: the dropped run must NOT replay from the journal.
	gateway.CloseEventJournal()
	restarted := NewGateway(cfg, nil)
	defer restarted.CloseEventJournal()
	restarted.taskEventsMu.Lock()
	_, replayed := restarted.taskEvents["drop"]
	restarted.taskEventsMu.Unlock()
	if replayed {
		t.Fatal("deleted run replayed from journal")
	}
}

func TestRunCancelProxiesToOrchestrator(t *testing.T) {
	received := make(chan map[string]any, 1)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		received <- body
		_ = json.NewEncoder(w).Encode(map[string]any{"cancelled": true, "run_id": body["run_id"]})
	}))
	defer upstream.Close()

	gateway := NewGateway(Config{OrchestratorURL: upstream.URL}, nil)
	server := httptest.NewServer(gateway.Handler())
	defer server.Close()

	response, err := http.Post(server.URL+"/api/runs/some-run/cancel", "application/json", nil)
	if err != nil || response.StatusCode != http.StatusOK {
		t.Fatalf("cancel failed: %v %v", err, response)
	}
	defer response.Body.Close()
	var body map[string]any
	_ = json.NewDecoder(response.Body).Decode(&body)
	if body["cancelled"] != true {
		t.Fatalf("cancel response wrong: %v", body)
	}
	select {
	case got := <-received:
		if got["run_id"] != "some-run" {
			t.Fatalf("orchestrator got wrong run id: %v", got)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("orchestrator never received the cancel")
	}
}

func TestWSBacklogSelectsRunByID(t *testing.T) {
	gateway := NewGateway(testConfig(), nil)
	// Activity order: older first, latest LAST (latestRun tracks recency).
	gateway.storeEvent(mustJSON(map[string]any{
		"event": "run.meta", "run_id": "older-run", "title": "older",
	}))
	gateway.storeEvent(mustJSON(map[string]any{
		"event": "run.meta", "run_id": "latest-run", "title": "latest",
	}))

	server := httptest.NewServer(gateway.Handler())
	defer server.Close()

	// No selection: latest run's backlog.
	wsURL := "ws" + strings.TrimPrefix(server.URL, "http") + "/ws"
	conn1, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
	if err != nil {
		t.Fatalf("ws dial failed: %v", err)
	}
	defer conn1.Close()
	first := readWSJSON(t, conn1)
	if first["run_id"] != "latest-run" {
		t.Fatalf("default backlog should be latest run, got %v", first["run_id"])
	}

	// Selection: the requested run's backlog.
	conn2, _, err := websocket.DefaultDialer.Dial(wsURL+"?run_id=older-run", nil)
	if err != nil {
		t.Fatalf("ws dial failed: %v", err)
	}
	defer conn2.Close()
	second := readWSJSON(t, conn2)
	if second["run_id"] != "older-run" {
		t.Fatalf("selected backlog should be older-run, got %v", second["run_id"])
	}
}

func TestCreateTaskEmitsRunMetaImmediately(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{"run_id": "x", "success": true, "outcome": "OK"})
	}))
	defer upstream.Close()

	gateway := NewGateway(Config{OrchestratorURL: upstream.URL}, nil)
	server := httptest.NewServer(gateway.Handler())
	defer server.Close()

	response, err := http.Post(server.URL+"/api/tasks", "application/json",
		strings.NewReader(`{"issue": "fix the parser bug please", "title": "Parser fix", "model_profile": "default"}`))
	if err != nil {
		t.Fatalf("create failed: %v", err)
	}
	defer response.Body.Close()

	gateway.taskEventsMu.Lock()
	events := gateway.taskEvents
	gateway.taskEventsMu.Unlock()
	for runID, stored := range events {
		var meta map[string]any
		if err := json.Unmarshal(stored[0], &meta); err != nil {
			continue
		}
		if meta["event"] != "run.meta" {
			t.Fatalf("first stored event must be run.meta, got %v (run %s)", meta["event"], runID)
		}
		if meta["title"] != "Parser fix" {
			t.Fatalf("custom title lost: %v", meta["title"])
		}
		if meta["model_profile"] != "default" {
			t.Fatalf("model profile lost: %v", meta["model_profile"])
		}
		return
	}
	t.Fatal("no run was stored at all")
}

// readWSJSON reads one WebSocket text message as JSON (3s deadline).
func readWSJSON(t *testing.T, conn *websocket.Conn) map[string]any {
	t.Helper()
	_ = conn.SetReadDeadline(time.Now().Add(3 * time.Second))
	_, raw, err := conn.ReadMessage()
	if err != nil {
		t.Fatalf("ws read failed: %v", err)
	}
	var payload map[string]any
	if err := json.Unmarshal(raw, &payload); err != nil {
		t.Fatalf("ws message not JSON: %v", err)
	}
	return payload
}
