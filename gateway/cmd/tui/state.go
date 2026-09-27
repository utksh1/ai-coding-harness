// Foreman cockpit TUI — model state (pure, no UI deps).
//
// State is the reducer over the cockpit event stream: it owns the agent
// org-tree (L1 architect → L2 manager → L3 specialists → L4 collaborators),
// the subtask board, the verification gates, the token meters and the raw
// event log. Nodes are created once and updated in place; the tree is never
// rebuilt from scratch per event.
package main

import (
	"fmt"
	"sort"
	"strings"
)

// AgentStatus is the per-agent status light in the org-tree.
type AgentStatus int

const (
	StatusIdle AgentStatus = iota
	StatusThinking
	StatusWorking
	StatusDone
	StatusFailed
)

// TaskStatus drives the subtask board.
type TaskStatus int

const (
	TaskPending TaskStatus = iota
	TaskRunning
	TaskDone
	TaskFailed
)

// StageOutcome is a verification-gate verdict.
type StageOutcome int

const (
	StagePending StageOutcome = iota
	StagePass
	StageFail
	StageSkip
)

// RunState is the headline verdict in the header.
const (
	RunRunning    = "RUNNING"
	RunVerified   = "VERIFIED"
	RunNotVerifed = "NOT VERIFIED"
	RunFailed     = "FAILED"
)

// canonicalStages is the six-gate order from docs/cockpit-events.md §6.
var canonicalStages = []string{
	"1-integrity", "2-self-check", "3-local-tests",
	"4-code-review", "5-security", "6-final-review",
}

// AgentNode is one agent in the org-tree. Children keep insertion order
// (roster order for L3, spawn order for collaborators).
type AgentNode struct {
	ID          string
	Role        string
	Level       int
	Model       string
	Specialties []string
	ToolTier    int
	Parent      string
	Children    []*AgentNode

	Status     AgentStatus
	Tokens     int // cumulative meter, SET from agent.usage.total_tokens_agent
	Step       int
	MaxSteps   int
	Phase      string
	Task       string // current task ("planning"/"coordination"/"st-N")
	LastTool   string
	LastArgs   string
	ToolCalls  []ToolCall
	Retries    int
	ReroutedTo string
	Summary    string

	// Assignment decoration (manager delegation).
	AssignedTask  string
	RoutingTotal  float64
	RoutingDetail string
}

// ToolCall is one agent.tool record for the Agent tab log.
type ToolCall struct {
	Step       int
	Task       string
	Tool       string
	ArgsDigest string
	OK         bool
	DurationMS int
	Result     string
}

// Task is one architect.plan subtask with live status.
type Task struct {
	ID         string
	Title      string
	Specialty  string
	Complexity int
	Files      []string
	DependsOn  []string
	Acceptance string
	RiskNotes  string

	Status        TaskStatus
	Agent         string // assigned executor
	CompletedBy   string // actual executor (differs after l2_reroute)
	Batch         int
	RoutingTotal  float64
	RoutingDetail string
	Retries       int
	ReroutedTo    string
	Summary       string
	Steps         int
	Tokens        int
}

// StageResult is one verification-gate verdict.
type StageResult struct {
	Name     string
	Outcome  StageOutcome
	Blocking bool
	Detail   string
	Seconds  float64
}

// ActivityEntry is one raw event kept for the Activity feed (formatted at
// render time so late-arriving roster facts, e.g. the manager id, resolve).
type ActivityEntry struct {
	Seq int
	Ev  Event
}

// State is the full cockpit model (one run at a time; run.start with a new
// run id resets run-scoped data but keeps the loaded roster).
type State struct {
	RunID     string
	RunState  string
	Outcome   string
	Error     string
	Issue     string
	Flags     []string
	ReproTest string
	Profile   *ProfileEvent

	GovernorMode     string
	TotalTokens      int
	PromptTokens     int
	CompletionTokens int

	Root         *AgentNode // architect (L1)
	agents       map[string]*AgentNode
	order        []string // first-seen order for stable iteration
	ManagerID    string
	RosterSource string // "api" | "derived"

	Tasks     map[string]*Task
	taskOrder []string

	Stages map[string]*StageResult

	Activity []ActivityEntry
	seq      int

	// Baseline summary.
	BaselineFailures int
	BaselineRunnable bool
}

// NewState builds an empty cockpit state.
func NewState() *State {
	return &State{
		agents:       make(map[string]*AgentNode),
		Tasks:        make(map[string]*Task),
		Stages:       make(map[string]*StageResult),
		RosterSource: "derived",
	}
}

// Root returns the architect node (nil before any roster/event).
func (s *State) Architect() *AgentNode { return s.Root }

// Agent looks an agent up (nil when unknown).
func (s *State) Agent(id string) *AgentNode { return s.agents[id] }

// Agents returns every agent node in first-seen order.
func (s *State) Agents() []*AgentNode {
	out := make([]*AgentNode, 0, len(s.order))
	for _, id := range s.order {
		if n := s.agents[id]; n != nil {
			out = append(out, n)
		}
	}
	return out
}

// FlatTree is the preorder walk used for arrow-key navigation.
func (s *State) FlatTree() []*AgentNode {
	var out []*AgentNode
	var walk func(n *AgentNode)
	walk = func(n *AgentNode) {
		if n == nil {
			return
		}
		out = append(out, n)
		for _, c := range n.Children {
			walk(c)
		}
	}
	walk(s.Root)
	return out
}

// TaskOrder returns task ids in plan order.
func (s *State) TaskOrder() []string { return s.taskOrder }

// StageOrder returns gates in canonical order first, unknown extras after.
func (s *State) StageOrder() []string {
	out := make([]string, 0, len(s.Stages))
	seen := make(map[string]bool, len(s.Stages))
	for _, name := range canonicalStages {
		if _, ok := s.Stages[name]; ok {
			out = append(out, name)
			seen[name] = true
		}
	}
	extras := make([]string, 0, len(s.Stages))
	for name := range s.Stages {
		if !seen[name] {
			extras = append(extras, name)
		}
	}
	sort.Strings(extras)
	return append(out, extras...)
}

// ApplyRoster installs the /api/agents roster (level/role driven). Agents
// already known from events keep their nodes; children are re-ordered to the
// roster's canonical order (event-derived newcomers keep relative order).
func (s *State) ApplyRoster(agents []RosterAgent) {
	s.RosterSource = "api"
	for _, ra := range agents {
		id := ra.AgentID
		if id == "" {
			id = ra.ID
		}
		if id == "" {
			continue
		}
		n := s.ensureAgent(id, ra.Role)
		if n == nil {
			continue
		}
		if ra.Model != "" {
			n.Model = ra.Model
		}
		if len(ra.Specialties) > 0 {
			n.Specialties = ra.Specialties
		}
		if ra.ToolTier > 0 {
			n.ToolTier = ra.ToolTier
		}
		if ra.Level > 0 {
			n.Level = ra.Level
		}
	}
	s.wireDeferred()

	// Canonical sibling order = roster order (rostered first, then
	// event-derived, stable within each group).
	idx := make(map[string]int, len(agents))
	for _, ra := range agents {
		id := ra.AgentID
		if id == "" {
			id = ra.ID
		}
		if id != "" {
			if _, dup := idx[id]; !dup {
				idx[id] = len(idx)
			}
		}
	}
	for _, n := range s.agents {
		children := n.Children
		sort.SliceStable(children, func(i, j int) bool {
			ci, iok := idx[children[i].ID]
			cj, jok := idx[children[j].ID]
			switch {
			case iok && jok:
				return ci < cj
			case iok != jok:
				return iok
			default:
				return false
			}
		})
	}
}

// ensureAgent returns the node for id, creating it (wired into the org-tree
// by role) when unseen. Existing nodes only get their role refreshed.
func (s *State) ensureAgent(id, role string) *AgentNode {
	if id == "" {
		return nil
	}
	if n, ok := s.agents[id]; ok {
		if role != "" && n.Role == "" {
			n.Role = role
		}
		return n
	}
	n := &AgentNode{ID: id, Role: role, Status: StatusIdle}
	s.agents[id] = n
	s.order = append(s.order, id)
	switch {
	case role == "architect":
		n.Level = 1
		if s.Root == nil {
			s.Root = n
			s.wireDeferred()
		} else if s.Root.ID != id {
			// A second architect stays a root-level sibling (degenerate,
			// still renderable).
			s.Root.Children = append(s.Root.Children, n)
			n.Parent = s.Root.ID
		}
	case role == "manager":
		n.Level = 2
		s.setManager(n)
	default:
		n.Level = 3
		s.attachSpecialist(n)
	}
	return n
}

// wireDeferred attaches any parentless nodes now that a root (or manager)
// exists — e.g. a specialist.assigned seen before the architect's first step.
func (s *State) wireDeferred() {
	if s.Root == nil {
		return
	}
	for _, id := range s.order {
		n := s.agents[id]
		if n == nil || n == s.Root || n.Parent != "" {
			continue
		}
		if n.Level == 2 {
			s.setManager(n)
		} else {
			s.attachSpecialist(n)
		}
	}
}

// setManager inserts the manager between the architect and the L3
// specialists (in-place re-parent, not a rebuild).
func (s *State) setManager(m *AgentNode) {
	if s.ManagerID == "" {
		s.ManagerID = m.ID
	}
	if s.Root == nil || s.Root.ID == m.ID {
		return
	}
	if m.Parent == "" {
		m.Parent = s.Root.ID
		s.Root.Children = append(s.Root.Children, m)
	}
	// Move specialists that currently hang directly off the architect.
	kept := s.Root.Children[:0]
	for _, c := range s.Root.Children {
		if c != m && c.Level >= 3 {
			s.reparent(c, m)
		} else {
			kept = append(kept, c)
		}
	}
	s.Root.Children = kept
}

// attachSpecialist hangs an L3 node under the manager when one is known,
// else directly under the architect (re-parented when a manager appears).
func (s *State) attachSpecialist(n *AgentNode) {
	parent := s.Root
	if m := s.agents[s.ManagerID]; m != nil {
		parent = m
	}
	if parent == nil {
		return // deferred: wired by wireDeferred on root/manager arrival
	}
	n.Parent = parent.ID
	parent.Children = append(parent.Children, n)
}

// reparent moves a node under a new parent (in place).
func (s *State) reparent(n, newParent *AgentNode) {
	if n.Parent == newParent.ID {
		return
	}
	if old := s.agents[n.Parent]; old != nil {
		kept := old.Children[:0]
		for _, c := range old.Children {
			if c != n {
				kept = append(kept, c)
			}
		}
		old.Children = kept
	}
	n.Parent = newParent.ID
	newParent.Children = append(newParent.Children, n)
}

// addCollaborator creates an L4 node attached to its "for" parent.
func (s *State) addCollaborator(id, forAgent, role string) *AgentNode {
	if id == "" {
		return nil
	}
	if n, ok := s.agents[id]; ok {
		return n
	}
	n := &AgentNode{ID: id, Role: role, Level: 4, Status: StatusIdle}
	s.agents[id] = n
	s.order = append(s.order, id)
	parent := s.agents[forAgent]
	if parent == nil {
		parent = s.ensureAgent(forAgent, "")
	}
	if parent != nil {
		n.Parent = parent.ID
		parent.Children = append(parent.Children, n)
	}
	return n
}

// resetRun clears run-scoped state for a fresh run.start (roster survives).
func (s *State) resetRun() {
	s.RunState = ""
	s.Outcome = ""
	s.Error = ""
	s.Issue = ""
	s.Flags = nil
	s.ReproTest = ""
	s.Profile = nil
	s.GovernorMode = ""
	s.TotalTokens = 0
	s.PromptTokens = 0
	s.CompletionTokens = 0
	s.Tasks = make(map[string]*Task)
	s.taskOrder = nil
	s.Stages = make(map[string]*StageResult)
	s.Activity = nil
	s.seq = 0
	s.BaselineFailures = 0
	s.BaselineRunnable = false

	for id, n := range s.agents {
		// Collaborators are run-scoped: drop them (with their subtrees).
		if n.Level >= 4 {
			if parent := s.agents[n.Parent]; parent != nil {
				kept := parent.Children[:0]
				for _, c := range parent.Children {
					if c != n {
						kept = append(kept, c)
					}
				}
				parent.Children = kept
			}
			delete(s.agents, id)
			s.order = removeString(s.order, id)
			continue
		}
		n.Status = StatusIdle
		n.Tokens = 0
		n.Step = 0
		n.MaxSteps = 0
		n.Phase = ""
		n.Task = ""
		n.LastTool = ""
		n.LastArgs = ""
		n.ToolCalls = nil
		n.Retries = 0
		n.ReroutedTo = ""
		n.Summary = ""
		n.AssignedTask = ""
		n.RoutingTotal = 0
		n.RoutingDetail = ""
	}
}

func removeString(xs []string, x string) []string {
	out := xs[:0]
	for _, v := range xs {
		if v != x {
			out = append(out, v)
		}
	}
	return out
}

// ensureTask returns (creating when needed) the task node for id.
func (s *State) ensureTask(id string) *Task {
	if id == "" {
		return nil
	}
	t, ok := s.Tasks[id]
	if !ok {
		t = &Task{ID: id}
		s.Tasks[id] = t
		s.taskOrder = append(s.taskOrder, id)
	}
	return t
}

// Apply folds one event into the state. Tolerance rule: every field access
// is guarded; unknown kinds only append to the activity log.
func (s *State) Apply(ev Event) {
	s.seq++
	defer func() {
		s.Activity = append(s.Activity, ActivityEntry{Seq: s.seq, Ev: ev})
	}()

	if ev.RunID != "" && ev.Kind == "run.start" && s.RunID != "" && ev.RunID != s.RunID {
		s.resetRun()
	}

	switch ev.Kind {
	case "run.start":
		if ev.RunID != "" {
			s.RunID = ev.RunID
		}
		s.RunState = RunRunning
		s.Issue = ev.Issue
		s.Flags = ev.Flags
		if a := s.Root; a != nil {
			a.Task = "planning"
		}

	case "architect.profile":
		s.Profile = ev.Profile
		if a := s.Root; a != nil && a.Task == "" {
			a.Task = "planning"
		}

	case "architect.plan":
		s.ReproTest = ev.ReproTest
		for _, spec := range ev.Subtasks {
			t := s.ensureTask(spec.ID)
			if t == nil {
				continue
			}
			t.Title = spec.Title
			t.Specialty = spec.Specialty
			t.Complexity = spec.Complexity
			t.Files = spec.Files
			t.DependsOn = spec.DependsOn
			t.Acceptance = spec.Acceptance
			t.RiskNotes = spec.RiskNotes
		}

	case "baseline.captured":
		s.BaselineFailures = ev.PreExistingFailures
		s.BaselineRunnable = ev.Runnable != nil && *ev.Runnable
		if s.ReproTest == "" {
			s.ReproTest = ev.ReproTest
		}

	case "specialist.assigned":
		if t := s.ensureTask(ev.Task); t != nil {
			t.Status = TaskRunning
			t.Agent = ev.Agent
			t.Batch = ev.Batch
			if ev.Routing != nil {
				t.RoutingTotal = ev.Routing.Total
				t.RoutingDetail = routingSummary(*ev.Routing)
			}
		}
		if n := s.ensureAgent(ev.Agent, ev.Role); n != nil {
			n.Status = StatusWorking
			n.AssignedTask = ev.Task
			if ev.Routing != nil {
				n.RoutingTotal = ev.Routing.Total
				n.RoutingDetail = routingSummary(*ev.Routing)
			}
		}

	case "specialist.result":
		if t := s.ensureTask(ev.Task); t != nil {
			if ev.Success != nil && *ev.Success {
				t.Status = TaskDone
			} else {
				t.Status = TaskFailed
			}
			t.CompletedBy = ev.Agent
			t.Summary = ev.Summary
			t.Steps = ev.Steps
			t.Tokens = ev.Tokens
		}
		if n := s.ensureAgent(ev.Agent, ev.Role); n != nil {
			if ev.Success != nil && *ev.Success {
				n.Status = StatusDone
			} else {
				n.Status = StatusFailed
			}
			n.Summary = ev.Summary
			// Meters are SET; result.tokens is a fallback when the agent
			// never emitted agent.usage.
			if n.Tokens == 0 && ev.Tokens > 0 {
				n.Tokens = ev.Tokens
			}
			if ev.Steps > 0 && n.Step == 0 {
				n.Step = ev.Steps
			}
		}

	case "specialist.collaborator_added":
		s.addCollaborator(ev.Agent, ev.For, ev.Role)

	case "agent.step":
		n := s.ensureAgent(ev.Agent, ev.Role)
		if n == nil {
			break
		}
		n.Step = ev.Step
		n.MaxSteps = ev.MaxSteps
		n.Phase = ev.Phase
		if ev.Task != "" {
			n.Task = ev.Task
		}
		switch ev.Phase {
		case "thinking":
			n.Status = StatusThinking
		case "responding":
			n.Status = StatusWorking
		}

	case "agent.tool":
		n := s.ensureAgent(ev.Agent, ev.Role)
		if n == nil {
			break
		}
		ok := ev.OK == nil || *ev.OK
		n.Status = StatusWorking
		n.LastTool = ev.Tool
		n.LastArgs = ev.ArgsDigest
		n.ToolCalls = append(n.ToolCalls, ToolCall{
			Step:       ev.Step,
			Task:       ev.Task,
			Tool:       ev.Tool,
			ArgsDigest: ev.ArgsDigest,
			OK:         ok,
			DurationMS: ev.DurationMS,
			Result:     ev.ResultDigest,
		})

	case "agent.usage":
		if n := s.ensureAgent(ev.Agent, ""); n != nil {
			if ev.TotalTokensAgent > 0 {
				n.Tokens = ev.TotalTokensAgent // SET, already cumulative
			} else if ev.Tokens > 0 && n.Tokens == 0 {
				n.Tokens = ev.Tokens
			} else if ev.TotalTokens > 0 && n.Tokens == 0 {
				n.Tokens = ev.TotalTokens
			}
		}

	case "recovery.l1_retry":
		if t := s.Tasks[ev.Task]; t != nil {
			t.Retries++
		}
		if n := s.agents[ev.Agent]; n != nil {
			n.Retries++
		}

	case "recovery.l2_guidance", "recovery.l2_reroute":
		if t := s.Tasks[ev.Task]; t != nil {
			t.ReroutedTo = ev.To
		}
		if ev.Kind == "recovery.l2_reroute" {
			if n := s.agents[ev.To]; n != nil {
				// The replacement takes over the live task.
				n.Status = StatusWorking
				n.AssignedTask = ev.Task
			}
		}

	case "recovery.l3_replan", "recovery.l3_skipped_budget":
		// Rendered via the activity feed; the board verdict arrives with
		// specialist.result / run.end.

	case "verification.stage":
		outcome := StagePending
		if ev.Passed != nil {
			outcome = StagePass
			if !*ev.Passed {
				outcome = StageFail
			}
		}
		// Gates skipped after a blocking failure report passed=true with a
		// detail that begins with "skipped" — surface them as SKIP, not PASS
		// (a gate that merely MENTIONS skips, e.g. "1 skipped", still passes).
		if ev.Skipped != nil && *ev.Skipped {
			outcome = StageSkip
		} else if outcome == StagePass && startsWithFold(ev.Detail, "skipped") {
			outcome = StageSkip
		}
		s.Stages[ev.Stage] = &StageResult{
			Name:     ev.Stage,
			Outcome:  outcome,
			Blocking: ev.Blocking != nil && *ev.Blocking,
			Detail:   ev.Detail,
			Seconds:  ev.DurationSeconds,
		}

	case "tokens.usage":
		if ev.Usage != nil {
			s.TotalTokens = ev.Usage.TotalTokens // SET, cumulative
			s.PromptTokens = ev.Usage.PromptTokens
			s.CompletionTokens = ev.Usage.CompletionTokens
		}
		if ev.GovernorMode != "" {
			s.GovernorMode = ev.GovernorMode
		}

	case "run.end":
		if ev.Success != nil && *ev.Success {
			s.RunState = RunVerified
		} else {
			s.RunState = RunNotVerifed
		}
		s.Outcome = ev.Outcome
		for _, n := range s.agents {
			if n.Status == StatusWorking || n.Status == StatusThinking {
				n.Status = StatusDone
			}
		}

	case "run.failed":
		s.RunState = RunFailed
		s.Error = ev.Error
		if ev.Stage != "" && s.Outcome == "" {
			s.Outcome = "failed at " + ev.Stage
		}
	}
}

// routingSummary renders the 40/20/20/20 breakdown compactly:
// specialty/availability/load/capability contributions.
func routingSummary(r Routing) string {
	return trimNum(r.Specialty) + "/" + trimNum(r.Availability) + "/" +
		trimNum(r.Load) + "/" + trimNum(r.Capability)
}

func trimNum(f float64) string {
	s := fmt.Sprintf("%.2f", f)
	s = strings.TrimRight(s, "0")
	s = strings.TrimSuffix(s, ".")
	if s == "" || s == "-" {
		return "0"
	}
	return s
}

func containsFold(s, sub string) bool {
	return strings.Contains(strings.ToLower(s), strings.ToLower(sub))
}

func startsWithFold(s, prefix string) bool {
	return strings.HasPrefix(strings.ToLower(s), strings.ToLower(prefix))
}
