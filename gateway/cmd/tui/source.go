// Foreman cockpit TUI — transport sources: the gateway WebSocket (live) and
// the JSONL replayer (offline verification mode).
package main

import (
	"bufio"
	"encoding/json"
	"net/http"
	"os"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/gorilla/websocket"
)

// wsFrame is one inbound WebSocket frame (or a reader error).
type wsFrame struct {
	raw string
	err error
}

// wsDial wraps the gorilla dialer (one indirection keeps update.go testable).
func wsDial(url string) (*websocket.Conn, *http.Response, error) {
	return websocket.DefaultDialer.Dial(url, nil)
}

// replayer drives offline replay of a cockpit-events fixture: one event per
// tick (default 150ms), pause/resume, single-step.
type replayer struct {
	events  []Event
	idx     int
	speed   time.Duration
	paused  bool
	skipped int // unparsable lines tolerated during load
}

// loadReplayer reads a .jsonl fixture line by line. Malformed lines are
// skipped (counted), never fatal.
func loadReplayer(path string, speed time.Duration) (*replayer, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := &replayer{speed: speed}
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		ev, err := parseEvent([]byte(line))
		if err != nil || ev.Kind == "" {
			r.skipped++
			continue
		}
		r.events = append(r.events, ev)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return r, nil
}

// next yields the next event (false when exhausted).
func (r *replayer) next() (Event, bool) {
	if r.idx >= len(r.events) {
		return Event{}, false
	}
	ev := r.events[r.idx]
	r.idx++
	return ev, true
}

func (r *replayer) done() bool { return r.idx >= len(r.events) }

func (r *replayer) total() int { return len(r.events) }

// ---------------------------------------------------------------------------
// Session management transport (platform P4): runs list, open-by-id, cancel,
// delete, projects, models — all plain HTTP against the gateway.

// RunInfo is one session row from GET /api/runs.
type RunInfo struct {
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

// ProjectInfo is one registered project from GET /api/projects.
type ProjectInfo struct {
	ID     string `json:"id"`
	Name   string `json:"name"`
	Path   string `json:"path"`
	Git    bool   `json:"git"`
	Branch string `json:"branch"`
	Dirty  bool   `json:"dirty"`
}

// ModelProfile is one models: entry from GET /api/models.
type ModelProfile struct {
	Profile  string `json:"profile"`
	Provider string `json:"provider"`
	Model    string `json:"model"`
}

// runsMsg carries the fetched session list (or the fetch error).
type runsMsg struct {
	runs []RunInfo
	err  error
}

// projectsMsg carries the registered project list.
type projectsMsg struct {
	projects []ProjectInfo
	err      error
}

// modelsMsg carries the available model profiles.
type modelsMsg struct {
	profiles []ModelProfile
	err      error
}

// runOpenedMsg carries one session's full event history (open-by-id).
type runOpenedMsg struct {
	runID  string
	events []Event
	err    error
}

// runActionMsg reports a cancel/delete outcome.
type runActionMsg struct {
	kind   string // "cancel" | "delete"
	runID  string
	detail string
	err    error
}

// fetchRunsCmd lists chat sessions.
func fetchRunsCmd(apiURL string) tea.Cmd {
	return func() tea.Msg {
		resp, err := httpClient.Get(apiURL + "/api/runs")
		if err != nil {
			return runsMsg{err: err}
		}
		defer resp.Body.Close()
		var body struct {
			Runs []RunInfo `json:"runs"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
			return runsMsg{err: err}
		}
		return runsMsg{runs: body.Runs}
	}
}

// fetchProjectsCmd lists registered projects (multi-project pickers).
func fetchProjectsCmd(apiURL string) tea.Cmd {
	return func() tea.Msg {
		resp, err := httpClient.Get(apiURL + "/api/projects")
		if err != nil {
			return projectsMsg{err: err}
		}
		defer resp.Body.Close()
		var body struct {
			Projects []ProjectInfo `json:"projects"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
			return projectsMsg{err: err}
		}
		return projectsMsg{projects: body.Projects}
	}
}

// fetchModelsCmd lists model profiles (the per-run model picker).
func fetchModelsCmd(apiURL string) tea.Cmd {
	return func() tea.Msg {
		resp, err := httpClient.Get(apiURL + "/api/models")
		if err != nil {
			return modelsMsg{err: err}
		}
		defer resp.Body.Close()
		var body struct {
			Profiles []ModelProfile `json:"profiles"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
			return modelsMsg{err: err}
		}
		return modelsMsg{profiles: body.Profiles}
	}
}

// openRunCmd fetches one session's stored event history (chat switching).
func openRunCmd(apiURL, runID string) tea.Cmd {
	return func() tea.Msg {
		resp, err := httpClient.Get(apiURL + "/api/tasks/" + runID)
		if err != nil {
			return runOpenedMsg{runID: runID, err: err}
		}
		defer resp.Body.Close()
		var body struct {
			Events []json.RawMessage `json:"events"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
			return runOpenedMsg{runID: runID, err: err}
		}
		events := make([]Event, 0, len(body.Events))
		for _, raw := range body.Events {
			if ev, err := parseEvent(raw); err == nil && ev.Kind != "" {
				events = append(events, ev)
			}
		}
		return runOpenedMsg{runID: runID, events: events}
	}
}

// cancelRunCmd is the chat stop button.
func cancelRunCmd(apiURL, runID string) tea.Cmd {
	return func() tea.Msg {
		resp, err := httpClient.Post(apiURL+"/api/runs/"+runID+"/cancel", "application/json", nil)
		if err != nil {
			return runActionMsg{kind: "cancel", runID: runID, err: err}
		}
		defer resp.Body.Close()
		var body map[string]any
		_ = json.NewDecoder(resp.Body).Decode(&body)
		detail, _ := body["detail"].(string)
		return runActionMsg{kind: "cancel", runID: runID, detail: detail}
	}
}

// deleteRunCmd removes a session from the chat list + journal.
func deleteRunCmd(apiURL, runID string) tea.Cmd {
	return func() tea.Msg {
		request, err := http.NewRequest(http.MethodDelete, apiURL+"/api/runs/"+runID, nil)
		if err != nil {
			return runActionMsg{kind: "delete", runID: runID, err: err}
		}
		resp, err := httpClient.Do(request)
		if err != nil {
			return runActionMsg{kind: "delete", runID: runID, err: err}
		}
		defer resp.Body.Close()
		return runActionMsg{kind: "delete", runID: runID}
	}
}
