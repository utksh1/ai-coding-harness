// Foreman cockpit TUI — unit tests for the pure state machine: both replay
// fixtures are folded through the parser/reducer and the org-tree shape,
// task statuses, token meters, gate results and the final verdict are
// asserted. Interactive rendering is deliberately untested.
package main

import (
	"bufio"
	"os"
	"strings"
	"testing"
	"time"
)

// applyFixture folds a .jsonl fixture through the pure state machine.
func applyFixture(t *testing.T, path string) *State {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open fixture: %v", err)
	}
	defer f.Close()

	st := NewState()
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		ev, err := parseEvent([]byte(line))
		if err != nil {
			t.Fatalf("parse %q: %v", line, err)
		}
		st.Apply(ev)
	}
	if err := scanner.Err(); err != nil {
		t.Fatalf("scan: %v", err)
	}
	return st
}

func childIDs(n *AgentNode) []string {
	ids := make([]string, 0, len(n.Children))
	for _, c := range n.Children {
		ids = append(ids, c.ID)
	}
	return ids
}

// ---------------------------------------------------------------------------
// Clean VERIFIED fixture (events.sample.jsonl)

func TestCleanFixtureTree(t *testing.T) {
	st := applyFixture(t, "../../web/events.sample.jsonl")

	if st.Root == nil || st.Root.ID != "architect-1" {
		t.Fatalf("root = %v, want architect-1", st.Root)
	}
	// No manager events in this fixture: derived tree hangs specialists
	// directly off the architect (in first-seen order).
	if got := childIDs(st.Root); got[0] != "impl-1" || got[1] != "ver-1" || len(got) != 2 {
		t.Fatalf("architect children = %v, want [impl-1 ver-1]", got)
	}
	if st.Agent("impl-1").Level != 3 || st.Agent("impl-1").Role != "implementer" {
		t.Fatalf("impl-1 = %+v", st.Agent("impl-1"))
	}
	if st.RosterSource != "derived" {
		t.Fatalf("roster source = %q, want derived", st.RosterSource)
	}
}

func TestCleanFixtureTaskBoard(t *testing.T) {
	st := applyFixture(t, "../../web/events.sample.jsonl")

	want := []struct {
		id     string
		status TaskStatus
		agent  string
		retry  int
	}{
		{"st-1", TaskDone, "impl-1", 1},
		{"st-2", TaskDone, "ver-1", 0},
		{"st-3", TaskDone, "impl-1", 0},
	}
	for _, w := range want {
		task := st.Tasks[w.id]
		if task == nil {
			t.Fatalf("task %s missing", w.id)
		}
		if task.Status != w.status || task.Agent != w.agent || task.Retries != w.retry {
			t.Fatalf("%s = status %d agent %q retries %d; want %d %q %d",
				w.id, task.Status, task.Agent, task.Retries, w.status, w.agent, w.retry)
		}
	}
	if got := st.TaskOrder(); strings.Join(got, ",") != "st-1,st-2,st-3" {
		t.Fatalf("task order = %v", got)
	}
	if st.Tasks["st-1"].Complexity != 6 || st.Tasks["st-1"].Specialty != "bugfix" {
		t.Fatalf("st-1 spec = %+v", st.Tasks["st-1"])
	}
	if len(st.Tasks["st-1"].Files) != 1 || st.Tasks["st-1"].Files[0] != "parse/core.py" {
		t.Fatalf("st-1 files = %v", st.Tasks["st-1"].Files)
	}
	if st.Tasks["st-1"].RoutingTotal != 0.8 {
		t.Fatalf("st-1 routing total = %v", st.Tasks["st-1"].RoutingTotal)
	}
}

func TestCleanFixtureTokensAndGates(t *testing.T) {
	st := applyFixture(t, "../../web/events.sample.jsonl")

	tokenWant := map[string]int{
		"architect-1": 25400, // 9400 then 16000 (SET semantics)
		"impl-1":      56300, // 9550 → 19450 → 32550 → 46990 → 56300
		"ver-1":       8580,
	}
	for id, want := range tokenWant {
		if got := st.Agent(id).Tokens; got != want {
			t.Fatalf("%s tokens = %d, want %d", id, got, want)
		}
	}
	if st.TotalTokens != 114000 {
		t.Fatalf("cumulative tokens = %d, want 114000 (set from tokens.usage)", st.TotalTokens)
	}

	if got := st.StageOrder(); len(got) != 6 || got[0] != "1-integrity" || got[5] != "6-final-review" {
		t.Fatalf("stage order = %v", got)
	}
	for _, name := range st.StageOrder() {
		if st.Stages[name].Outcome != StagePass {
			t.Fatalf("stage %s = %d, want PASS", name, st.Stages[name].Outcome)
		}
	}
	if st.RunState != RunVerified {
		t.Fatalf("run state = %q, want VERIFIED", st.RunState)
	}
	if st.ReproTest != "tests/test_width_alignment.py" {
		t.Fatalf("repro = %q", st.ReproTest)
	}
	if st.GovernorMode != "NORMAL" {
		t.Fatalf("governor = %q", st.GovernorMode)
	}
}

// ---------------------------------------------------------------------------
// Collaborator fixture (events.sample-collab.jsonl): manager + L4 nesting,
// retries, reroute, failing gate, NOT VERIFIED.

func TestCollabFixtureTree(t *testing.T) {
	st := applyFixture(t, "../../web/events.sample-collab.jsonl")

	if st.Root == nil || st.Root.ID != "architect-1" {
		t.Fatalf("root = %v", st.Root)
	}
	if got := childIDs(st.Root); len(got) != 1 || got[0] != "mgr-1" {
		t.Fatalf("architect children = %v, want [mgr-1]", got)
	}
	mgr := st.Agent("mgr-1")
	if mgr.Level != 2 || mgr.Parent != "architect-1" {
		t.Fatalf("mgr-1 = level %d parent %q", mgr.Level, mgr.Parent)
	}
	if got := childIDs(mgr); len(got) != 2 || got[0] != "impl-1" || got[1] != "ver-1" {
		t.Fatalf("manager children = %v, want [impl-1 ver-1]", got)
	}
	impl := st.Agent("impl-1")
	if impl.Parent != "mgr-1" {
		t.Fatalf("impl-1 parent = %q, want mgr-1", impl.Parent)
	}
	if got := childIDs(impl); len(got) != 1 || got[0] != "impl-1-collab-1" {
		t.Fatalf("impl-1 children = %v, want [impl-1-collab-1]", got)
	}
	collab := st.Agent("impl-1-collab-1")
	if collab.Level != 4 || collab.Parent != "impl-1" || collab.Role != "implementer" {
		t.Fatalf("collab = %+v", collab)
	}
}

func TestCollabFixtureTasksAndRecovery(t *testing.T) {
	st := applyFixture(t, "../../web/events.sample-collab.jsonl")

	st1 := st.Tasks["st-1"]
	if st1 == nil {
		t.Fatal("st-1 missing")
	}
	if st1.Retries != 2 {
		t.Fatalf("st-1 retries = %d, want 2", st1.Retries)
	}
	if st1.ReroutedTo != "impl-1-collab-1" {
		t.Fatalf("st-1 reroutedTo = %q", st1.ReroutedTo)
	}
	if st1.CompletedBy != "impl-1-collab-1" || st1.Agent != "impl-1" {
		t.Fatalf("st-1 assigned %q completed by %q", st1.Agent, st1.CompletedBy)
	}
	if st1.Status != TaskDone {
		t.Fatalf("st-1 status = %d, want DONE", st1.Status)
	}
	if st.Tasks["st-2"].Status != TaskDone || st.Tasks["st-2"].Agent != "ver-1" {
		t.Fatalf("st-2 = %+v", st.Tasks["st-2"])
	}
	if st.Agent("impl-1").Retries != 2 {
		t.Fatalf("impl-1 retries = %d, want 2", st.Agent("impl-1").Retries)
	}
}

func TestCollabFixtureTokensGatesVerdict(t *testing.T) {
	st := applyFixture(t, "../../web/events.sample-collab.jsonl")

	tokenWant := map[string]int{
		"architect-1":     9750,
		"mgr-1":           5380,
		"impl-1":          26100,
		"impl-1-collab-1": 13700,
		"ver-1":           14100, // fallback: no agent.usage, set from specialist.result
	}
	for id, want := range tokenWant {
		if got := st.Agent(id).Tokens; got != want {
			t.Fatalf("%s tokens = %d, want %d", id, got, want)
		}
	}
	if st.TotalTokens != 84600 {
		t.Fatalf("cumulative tokens = %d, want 84600", st.TotalTokens)
	}

	gateWant := map[string]StageOutcome{
		"1-integrity":    StagePass,
		"2-self-check":   StagePass,
		"3-local-tests":  StageFail,
		"4-code-review":  StageSkip, // "skipped after blocking failure"
		"5-security":     StageSkip,
		"6-final-review": StageSkip,
	}
	for name, want := range gateWant {
		got := st.Stages[name]
		if got == nil || got.Outcome != want {
			t.Fatalf("gate %s = %+v, want outcome %d", name, got, want)
		}
	}
	if st.RunState != RunNotVerifed {
		t.Fatalf("run state = %q, want NOT VERIFIED", st.RunState)
	}
}

// ---------------------------------------------------------------------------
// Roster + events: the canonical live-mode shape (architect → manager →
// specialists → collaborators) even when events arrive out of order.

func TestRosterThenEventsTree(t *testing.T) {
	st := NewState()
	st.ApplyRoster([]RosterAgent{
		{AgentID: "architect-1", Role: "architect", Specialties: []string{"architecture", "code-review"}, ToolTier: 3, Level: 1},
		{AgentID: "mgr-1", Role: "manager", Specialties: []string{"coordination"}, ToolTier: 1, Level: 2},
		{AgentID: "locator-1", Role: "locator", Specialties: []string{"localization"}, ToolTier: 1, Level: 3},
		{AgentID: "impl-1", Role: "implementer", Specialties: []string{"backend-api"}, ToolTier: 3, Level: 3},
		{AgentID: "ver-1", Role: "verifier", Specialties: []string{"testing"}, ToolTier: 3, Level: 3},
	})
	st.ApplyRoster(nil) // idempotent no-op

	if got := childIDs(st.Root); len(got) != 1 || got[0] != "mgr-1" {
		t.Fatalf("architect children = %v, want [mgr-1]", got)
	}
	mgr := st.Agent("mgr-1")
	if got := childIDs(mgr); len(got) != 3 || got[0] != "locator-1" || got[1] != "impl-1" || got[2] != "ver-1" {
		t.Fatalf("manager children = %v, want roster order", got)
	}
	if st.Agent("impl-1").Specialties[0] != "backend-api" || st.Agent("impl-1").ToolTier != 3 {
		t.Fatalf("roster fields lost: %+v", st.Agent("impl-1"))
	}

	// Feed the collab fixture on top: collaborator must nest under impl-1.
	st2 := applyFixture(t, "../../web/events.sample-collab.jsonl")
	st2.ApplyRoster([]RosterAgent{
		{AgentID: "architect-1", Role: "architect", Level: 1},
		{AgentID: "mgr-1", Role: "manager", Level: 2},
		{AgentID: "locator-1", Role: "locator", Level: 3},
		{AgentID: "impl-1", Role: "implementer", Level: 3},
		{AgentID: "ver-1", Role: "verifier", Level: 3},
	})
	if got := childIDs(st2.Agent("mgr-1")); len(got) != 3 || got[0] != "locator-1" {
		t.Fatalf("manager children = %v, want [locator-1 impl-1 ver-1]", got)
	}
	if got := childIDs(st2.Agent("impl-1")); len(got) != 1 || got[0] != "impl-1-collab-1" {
		t.Fatalf("impl-1 children = %v, want [impl-1-collab-1]", got)
	}
}

// ---------------------------------------------------------------------------
// Event tolerance: garbage, unknown kinds, v1 string subtasks, missing fields.

func TestTolerance(t *testing.T) {
	st := NewState()

	// Empty object and unknown kind never crash.
	for _, raw := range []string{
		`{}`,
		`{"event": "mystery.kind", "run_id": "abc123"}`,
		`{"event": "agent.tool", "agent": ""}`,
		`{"event": "run.end"}`,
		`{"event": "verification.stage"}`,
		`{"event": "specialist.assigned"}`,
		`{"event": "recovery.l1_retry"}`,
	} {
		ev, err := parseEvent([]byte(raw))
		if err != nil {
			t.Fatalf("parse %q: %v", raw, err)
		}
		st.Apply(ev) // must not panic
	}

	// Garbage JSON is a parse error (caller skips), never a crash.
	if _, err := parseEvent([]byte(`not json`)); err == nil {
		t.Fatal("garbage line should fail to parse")
	}

	// Unknown kinds are kept in the activity log (rendered generically).
	last := st.Activity[len(st.Activity)-1]
	if last.Ev.Kind != "recovery.l1_retry" {
		t.Fatalf("activity kept = %q", last.Ev.Kind)
	}
	unknown := FormatActivity(Event{Kind: "mystery.kind", RunID: "abc123"}, st)
	if unknown.Line != "«mystery.kind» · run abc123" {
		t.Fatalf("generic line = %q", unknown.Line)
	}

	// v1 bare-string subtasks decode as id-only tasks.
	st2 := NewState()
	st2.Apply(Event{Kind: "architect.plan", Subtasks: SubtaskList{{ID: "st-9"}, {ID: "st-8"}}})
	if got := st2.TaskOrder(); strings.Join(got, ",") != "st-9,st-8" {
		t.Fatalf("v1 subtasks = %v", got)
	}
	if st2.Tasks["st-9"].Status != TaskPending {
		t.Fatal("v1 subtask should be PENDING")
	}

	// Malformed subtask array entries are skipped, not fatal.
	var list SubtaskList
	if err := list.UnmarshalJSON([]byte(`[{"id":"ok"}, 5, "x"]`)); err != nil {
		t.Fatalf("mixed subtasks: %v", err)
	}
	if len(list) != 2 || list[0].ID != "ok" || list[1].ID != "x" {
		t.Fatalf("mixed subtasks = %+v", list)
	}

	// agent.usage without total_tokens_agent falls back to total_tokens.
	st4 := NewState()
	st4.Apply(Event{Kind: "agent.usage", Agent: "impl-1", TotalTokens: 777})
	if st4.Agent("impl-1").Tokens != 777 {
		t.Fatalf("fallback tokens = %d", st4.Agent("impl-1").Tokens)
	}
}

// ---------------------------------------------------------------------------
// Run reset: a second run.start clears run-scoped state, keeps the roster.

func TestResetRunKeepsRoster(t *testing.T) {
	st := applyFixture(t, "../../web/events.sample-collab.jsonl")
	st.ApplyRoster([]RosterAgent{{AgentID: "locator-1", Role: "locator", Level: 3}})

	st.Apply(Event{Kind: "run.start", RunID: "newrun9999", Issue: "second task"})

	if st.RunID != "newrun9999" {
		t.Fatalf("run id = %q", st.RunID)
	}
	if len(st.Tasks) != 0 || len(st.Stages) != 0 || st.TotalTokens != 0 {
		t.Fatalf("run-scoped state not cleared: %d tasks, %d stages, %d tokens",
			len(st.Tasks), len(st.Stages), st.TotalTokens)
	}
	if st.Agent("impl-1-collab-1") != nil {
		t.Fatal("run-scoped collaborator survived the reset")
	}
	if st.Agent("impl-1") == nil || st.Agent("mgr-1") == nil || st.Agent("locator-1") == nil {
		t.Fatal("roster agents lost on reset")
	}
	if st.Agent("impl-1").Tokens != 0 {
		t.Fatalf("agent meters not reset: %d", st.Agent("impl-1").Tokens)
	}
}

// ---------------------------------------------------------------------------
// Replayer loader.

func TestReplayer(t *testing.T) {
	r, err := loadReplayer("../../web/events.sample.jsonl", 0)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if r.total() != 51 {
		t.Fatalf("total = %d, want 51", r.total())
	}
	for i := 0; i < 51; i++ {
		if _, ok := r.next(); !ok {
			t.Fatalf("event %d missing", i)
		}
	}
	if !r.done() {
		t.Fatal("replayer should be exhausted")
	}
	if _, ok := r.next(); ok {
		t.Fatal("next() past the end should be false")
	}

	// Tolerant loading: garbage lines are skipped and counted.
	path := t.TempDir() + "/mixed.jsonl"
	content := "{\"event\":\"run.start\",\"run_id\":\"r1\"}\n\nbogus line\n{\"event\":\"run.end\",\"run_id\":\"r1\"}\n"
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	r2, err := loadReplayer(path, 150*time.Millisecond)
	if err != nil {
		t.Fatalf("mixed load: %v", err)
	}
	if r2.total() != 2 || r2.skipped != 1 {
		t.Fatalf("mixed fixture: total=%d skipped=%d", r2.total(), r2.skipped)
	}
}

func TestThousandsAndFmtTokens(t *testing.T) {
	cases := []struct {
		in   int
		want string
	}{
		{114000, "114,000"}, {999, "999"}, {0, "0"}, {-42, "-42"},
	}
	for _, c := range cases {
		if got := thousands(c.in); got != c.want {
			t.Fatalf("thousands(%d) = %q, want %q", c.in, got, c.want)
		}
	}
	short := []struct {
		in   int
		want string
	}{
		{25400, "25k"}, {9400, "9.4k"}, {999, "999"}, {1234567, "1.2M"},
	}
	for _, c := range short {
		if got := fmtTokens(c.in); got != c.want {
			t.Fatalf("fmtTokens(%d) = %q, want %q", c.in, got, c.want)
		}
	}
}
