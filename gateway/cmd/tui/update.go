// Foreman cockpit TUI — the bubbletea model: focus/panes/tabs, keybindings,
// the new-task modal, replay ticks and transport message handling.
package main

import (
	"encoding/json"
	"net/http"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
)

// tab identifies the main pane's tab (1-5 in the footer).
type tab int

const (
	tabPlan tab = iota
	tabAgent
	tabActivity
	tabVerify
	tabDiff
)

var tabNames = []string{"Plan", "Agent", "Activity", "Verify", "Diff"}

// focus is which pane owns the arrow keys.
type focus int

const (
	focusMain focus = iota
	focusTree
)

// model is the cockpit UI model; the event-derived world lives in
// m.state (pure, unit-tested).
type model struct {
	state *State

	// Transport (live mode).
	wsURL      string
	apiURL     string
	connected  bool
	connDetail string
	msgs       chan wsFrame

	// Replay mode (nil in live mode).
	replay *replayer

	// UI state.
	activeTab      tab
	focus          focus
	cursor         int // index into state.FlatTree()
	selectedAgent  string
	activityOffset int // lines scrolled back from the tail
	agentLogOffset int // lines scrolled back from the tail
	diffOffset     int // lines scrolled forward from the head
	width, height  int

	// New-task modal.
	inputMode  bool
	inputField int // 0: issue, 1: repo_root
	issueInput string
	repoInput  string
	notice     string

	// Evidence fetch (Diff tab).
	diffStatus     string // "", "fetching", "none", "error", "ok"
	diffErr        string
	diffLines      []string
	diffFetchedRun string

	headless bool // non-tty stdin: auto-quit when a replay finishes
}

// ---------------------------------------------------------------------------
// Bubbletea plumbing

type frameMsg struct{ raw string }
type connStateMsg struct {
	connected bool
	detail    string
}
type rosterMsg struct {
	agents []RosterAgent
	err    error
}
type taskSubmittedMsg struct {
	runID string
	err   error
}
type diffMsg struct {
	runID string
	found bool
	err   error
}
type replayTickMsg struct{}
type autoReconnMsg struct{}
type quitNowMsg struct{}
type replayKickMsg struct{}

// Init boots the transport (live) or the animation (replay).
func (m model) Init() tea.Cmd {
	if m.replay != nil {
		if m.replay.done() {
			// Even in instant mode the roster fetch is worth firing: the
			// org tree wants the idle agents (manager, unused specialists)
			// that emit no events. Best-effort - failure falls back to the
			// event-derived tree.
			cmd := fetchRosterCmd(m.apiURL)
			if m.headless {
				return tea.Batch(cmd, delayedQuit())
			}
			return cmd
		}
		return tea.Batch(
			func() tea.Msg { return replayKickMsg{} },
			fetchRosterCmd(m.apiURL),
		)
	}
	return tea.Batch(dialCmd(m), fetchRosterCmd(m.apiURL))
}

func delayedQuit() tea.Cmd {
	return tea.Tick(150*time.Millisecond, func(time.Time) tea.Msg { return quitNowMsg{} })
}

// Update handles every message; run-scope mutations go through State.Apply.
func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width, m.height = msg.Width, msg.Height
		return m, nil

	case quitNowMsg:
		return m, tea.Quit

	case replayKickMsg:
		if m.replay == nil || m.replay.done() {
			return m, nil
		}
		if m.replay.speed <= 0 {
			// Instant mode: fold everything in, then settle.
			for {
				ev, ok := m.replay.next()
				if !ok {
					break
				}
				m.state.Apply(ev)
			}
			if m.headless {
				return m, delayedQuit()
			}
			return m, nil
		}
		return m, tickReplay(m.replay.speed)

	case replayTickMsg:
		if m.replay == nil || m.replay.paused || m.replay.done() {
			return m, nil
		}
		if ev, ok := m.replay.next(); ok {
			m.state.Apply(ev)
		}
		if m.replay.done() {
			if m.headless {
				return m, delayedQuit()
			}
			return m, nil
		}
		return m, tickReplay(m.replay.speed)

	case rosterMsg:
		if msg.err != nil {
			m.state.RosterSource = "derived"
			return m, nil
		}
		m.state.ApplyRoster(msg.agents)
		return m, nil

	case connStateMsg:
		m.connected = msg.connected
		m.connDetail = msg.detail
		if msg.connected {
			return m, listenCmd(m.msgs)
		}
		if m.replay == nil {
			// Gentle auto-reconnect; [r] forces it immediately.
			return m, tea.Tick(5*time.Second, func(time.Time) tea.Msg { return autoReconnMsg{} })
		}
		return m, nil

	case autoReconnMsg:
		if !m.connected && m.replay == nil {
			m.msgs = make(chan wsFrame, 256)
			return m, dialCmd(m)
		}
		return m, nil

	case frameMsg:
		if ev, err := parseEvent([]byte(msg.raw)); err == nil {
			wasRunning := m.state.RunState == RunRunning
			m.state.Apply(ev)
			// Evidence (patch.diff) loads when the run ends.
			if wasRunning && m.state.RunState != RunRunning &&
				m.state.RunState != "" && m.replay == nil &&
				m.diffFetchedRun != m.state.RunID && m.state.RunID != "" {
				m.diffFetchedRun = m.state.RunID
				m.diffStatus = "fetching"
				return m, tea.Batch(listenCmd(m.msgs), fetchDiffCmd(m.apiURL, m.state.RunID))
			}
		} else {
			// Transport garbage: never crash — count it as a raw line.
			m.notice = "unparsable frame received (skipped)"
		}
		return m, listenCmd(m.msgs)

	case taskSubmittedMsg:
		if msg.err != nil {
			m.notice = "launch failed: " + msg.err.Error()
			m.inputMode = false
			return m, nil
		}
		m.notice = "task launched · run " + shortRun(msg.runID)
		m.inputMode = false
		if msg.runID != "" {
			m.state.RunID = msg.runID
			m.state.RunState = RunRunning
		}
		return m, nil

	case diffMsg:
		if msg.err != nil {
			m.diffStatus = "error"
			m.diffErr = msg.err.Error()
			return m, nil
		}
		if !msg.found {
			m.diffStatus = "none"
			return m, nil
		}
		return m, nil // content arrives via contentDiffMsg

	case contentDiffMsg:
		m.diffLines = splitDiff(msg.content)
		m.diffStatus = "ok"
		m.diffOffset = 0
		return m, nil

	case tea.KeyMsg:
		if m.inputMode {
			return m.updateInputModal(msg)
		}
		return m.updateKeys(msg)
	}
	return m, nil
}

// updateKeys is the normal (non-modal) keymap.
func (m model) updateKeys(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	key := msg.String()
	switch key {
	case "q", "ctrl+c":
		return m, tea.Quit

	case "1", "2", "3", "4", "5":
		m.activeTab = tab(runeToInt(key))
		m.focus = focusMain
		return m, nil

	case "tab":
		if m.focus == focusTree {
			m.focus = focusMain
		} else {
			m.focus = focusTree
		}
		return m, nil

	case "t":
		m.focus = focusTree
		return m, nil

	case "a":
		if id := m.cursorAgentID(); id != "" {
			m.selectedAgent = id
		}
		m.activeTab = tabAgent
		return m, nil

	case "up", "k":
		if m.focus == focusTree {
			m.moveCursor(-1)
		} else {
			m.scrollActive(-1)
		}
		return m, nil

	case "down", "j":
		if m.focus == focusTree {
			m.moveCursor(1)
		} else {
			m.scrollActive(1)
		}
		return m, nil

	case "g", "home":
		if m.focus == focusTree {
			m.cursor = 0
		} else if m.activeTab == tabDiff {
			m.diffOffset = 0
		} else {
			m.activityOffset, m.agentLogOffset = 1<<30, 1<<30 // top
		}
		return m, nil

	case "G", "end":
		if m.focus == focusTree {
			if n := len(m.state.FlatTree()); n > 0 {
				m.cursor = n - 1
			}
		} else if m.activeTab == tabDiff {
			m.diffOffset = 1 << 30
		} else {
			m.activityOffset, m.agentLogOffset = 0, 0 // tail
		}
		return m, nil

	case "enter":
		if m.focus == focusTree {
			if id := m.cursorAgentID(); id != "" {
				m.selectedAgent = id
				m.activeTab = tabAgent
				m.agentLogOffset = 0
			}
			return m, nil
		}
		return m, nil

	case "n":
		if m.replay == nil {
			m.inputMode = true
			m.inputField = 0
			m.notice = ""
		} else {
			m.notice = "task launch is disabled in replay mode"
		}
		return m, nil

	case "r":
		if m.replay == nil {
			m.connected = false
			m.connDetail = "reconnecting…"
			m.msgs = make(chan wsFrame, 256)
			return m, dialCmd(m)
		}
		// Replay: r resets nothing (r reconnect is live-only).
		return m, nil

	case " ":
		if m.replay != nil {
			m.replay.paused = !m.replay.paused
			if !m.replay.paused && !m.replay.done() {
				return m, tickReplay(m.replay.speed)
			}
		}
		return m, nil

	case "right":
		if m.replay != nil && !m.replay.done() {
			m.replay.paused = true
			if ev, ok := m.replay.next(); ok {
				m.state.Apply(ev)
			}
		}
		return m, nil
	}
	return m, nil
}

// updateInputModal is the new-task modal keymap ([tab] field, [enter] run,
// [esc] cancel, printable input).
func (m model) updateInputModal(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.String() {
	case "esc":
		m.inputMode = false
		m.notice = "task entry cancelled"
		return m, nil
	case "tab":
		m.inputField = (m.inputField + 1) % 2
		return m, nil
	case "enter":
		if strings.TrimSpace(m.issueInput) == "" {
			m.notice = "issue description required"
			return m, nil
		}
		m.notice = "launching run…"
		return m, submitTaskCmd(m.apiURL, m.issueInput, m.repoInput)
	case "backspace":
		if m.inputField == 0 {
			m.issueInput = dropLastRune(m.issueInput)
		} else {
			m.repoInput = dropLastRune(m.repoInput)
		}
		return m, nil
	default:
		for _, r := range msg.Runes {
			if r >= 0x20 && r != 0x7f {
				if m.inputField == 0 {
					m.issueInput += string(r)
				} else {
					m.repoInput += string(r)
				}
			}
		}
		return m, nil
	}
}

func dropLastRune(s string) string {
	rs := []rune(s)
	if len(rs) == 0 {
		return s
	}
	return string(rs[:len(rs)-1])
}

func runeToInt(key string) int {
	if len(key) != 1 {
		return 0
	}
	return int(key[0] - '1')
}

// moveCursor moves the tree selection cursor with clamping.
func (m *model) moveCursor(d int) {
	n := len(m.state.FlatTree())
	if n == 0 {
		m.cursor = 0
		return
	}
	m.cursor += d
	if m.cursor < 0 {
		m.cursor = 0
	}
	if m.cursor >= n {
		m.cursor = n - 1
	}
}

// cursorAgentID returns the id of the tree node under the cursor.
func (m model) cursorAgentID() string {
	flat := m.state.FlatTree()
	if m.cursor < 0 || m.cursor >= len(flat) {
		return ""
	}
	return flat[m.cursor].ID
}

// scrollActive scrolls the active main-pane tab.
func (m *model) scrollActive(d int) {
	switch m.activeTab {
	case tabActivity:
		m.activityOffset += d
		if m.activityOffset < 0 {
			m.activityOffset = 0
		}
	case tabAgent:
		m.agentLogOffset += d
		if m.agentLogOffset < 0 {
			m.agentLogOffset = 0
		}
	case tabDiff:
		m.diffOffset += d
		if m.diffOffset < 0 {
			m.diffOffset = 0
		}
	}
}

func tickReplay(speed time.Duration) tea.Cmd {
	return tea.Tick(speed, func(time.Time) tea.Msg { return replayTickMsg{} })
}

// ---------------------------------------------------------------------------
// Transport commands (live mode)

func dialCmd(m model) tea.Cmd {
	msgs := m.msgs
	wsURL := m.wsURL
	return func() tea.Msg {
		conn, _, err := wsDial(wsURL)
		if err != nil {
			return connStateMsg{connected: false, detail: err.Error()}
		}
		go func() {
			defer conn.Close()
			for {
				_, payload, err := conn.ReadMessage()
				if err != nil {
					select {
					case msgs <- wsFrame{err: err}:
					default:
					}
					return
				}
				select {
				case msgs <- wsFrame{raw: string(payload)}:
				default: // drop frames when the UI stalls (feed caps anyway)
				}
			}
		}()
		return connStateMsg{connected: true, detail: wsURL}
	}
}

func listenCmd(msgs chan wsFrame) tea.Cmd {
	return func() tea.Msg {
		f := <-msgs
		if f.err != nil {
			return connStateMsg{connected: false, detail: f.err.Error()}
		}
		return frameMsg{raw: f.raw}
	}
}

func fetchRosterCmd(apiURL string) tea.Cmd {
	return func() tea.Msg {
		resp, err := httpClient.Get(apiURL + "/api/agents")
		if err != nil {
			return rosterMsg{err: err}
		}
		defer resp.Body.Close()
		var rr RosterResponse
		if err := json.NewDecoder(resp.Body).Decode(&rr); err != nil {
			return rosterMsg{err: err}
		}
		return rosterMsg{agents: rr.Agents}
	}
}

func submitTaskCmd(apiURL, issue, repoRoot string) tea.Cmd {
	return func() tea.Msg {
		body, _ := json.Marshal(map[string]string{
			"issue":     issue,
			"repo_root": repoRoot,
		})
		resp, err := httpClient.Post(apiURL+"/api/tasks", "application/json", strings.NewReader(string(body)))
		if err != nil {
			return taskSubmittedMsg{err: err}
		}
		defer resp.Body.Close()
		var res struct {
			RunID string `json:"run_id"`
		}
		_ = json.NewDecoder(resp.Body).Decode(&res)
		return taskSubmittedMsg{runID: res.RunID}
	}
}

type contentDiffMsg struct {
	content string
}

func fetchDiffCmd(apiURL, runID string) tea.Cmd {
	return func() tea.Msg {
		resp, err := httpClient.Get(apiURL + "/api/evidence/" + runID + "/file/patch.diff")
		if err != nil {
			return diffMsg{runID: runID, err: err}
		}
		defer resp.Body.Close()
		var ev EvidenceResponse
		if err := json.NewDecoder(resp.Body).Decode(&ev); err != nil {
			return diffMsg{runID: runID, err: err}
		}
		if !ev.Found {
			return diffMsg{runID: runID, found: false}
		}
		return contentDiffMsg{content: ev.Content}
	}
}

var httpClient = &http.Client{Timeout: 8 * time.Second}

func splitDiff(content string) []string {
	content = strings.TrimRight(content, "\n")
	if content == "" {
		return nil
	}
	return strings.Split(content, "\n")
}
