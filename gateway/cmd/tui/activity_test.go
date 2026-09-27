// Foreman cockpit TUI — unit tests for the human-readable activity formatter.
package main

import (
	"strings"
	"testing"
)

// collabState returns the fully-reduced collaborator fixture (manager mgr-1
// known, so delegation lines can name the manager).
func collabState(t *testing.T) *State {
	t.Helper()
	return applyFixture(t, "../../web/events.sample-collab.jsonl")
}

func TestFormatActivityToolCall(t *testing.T) {
	st := collabState(t)
	ev := Event{
		Kind: "agent.tool", RunID: "f7e6d5c4b3a2", Agent: "impl-1", Role: "implementer",
		Task: "st-1", Step: 2, Tool: "read_file", ArgsDigest: "src/validator.py",
		OK: boolPtr(true), DurationMS: 15, ResultDigest: "912 lines",
	}
	got := FormatActivity(ev, st)
	if got.Line != "impl-1 → st-1: read src/validator.py (15ms)" {
		t.Fatalf("tool line = %q", got.Line)
	}
	if got.Class != classNormal {
		t.Fatalf("class = %d", got.Class)
	}
}

func TestFormatActivityToolFailure(t *testing.T) {
	st := collabState(t)
	ev := Event{
		Kind: "agent.tool", Agent: "impl-1", Task: "st-1", Tool: "apply_edit",
		ArgsDigest: "src/validator.py (+120/-380)", OK: boolPtr(false),
		DurationMS: 30, ResultDigest: "context mismatch at line 440",
	}
	got := FormatActivity(ev, st)
	want := "impl-1 → st-1: edit src/validator.py (+120/-380) (30ms) ✗ context mismatch at line 440"
	if got.Line != want {
		t.Fatalf("failed tool line = %q, want %q", got.Line, want)
	}
	if got.Class != classErr {
		t.Fatalf("class = %d, want classErr", got.Class)
	}
}

func TestFormatActivityRouting(t *testing.T) {
	st := collabState(t)
	ev := Event{
		Kind: "specialist.assigned", Task: "st-1", Agent: "impl-1", Role: "implementer",
		Batch: 1, Routing: &Routing{Specialty: 0.4, Availability: 0.2, Load: 0.2, Capability: 0.2, Total: 0.8},
	}
	got := FormatActivity(ev, st)
	if got.Line != "mgr-1 routed st-1 → impl-1 (0.80)" {
		t.Fatalf("routing line = %q", got.Line)
	}

	// Without any manager knowledge the line degrades to "mgr".
	fresh := NewState()
	got = FormatActivity(ev, fresh)
	if got.Line != "mgr routed st-1 → impl-1 (0.80)" {
		t.Fatalf("routing line (no manager) = %q", got.Line)
	}

	// Missing routing breakdown: score renders as 0.00, no crash.
	ev.Routing = nil
	got = FormatActivity(ev, st)
	if got.Line != "mgr-1 routed st-1 → impl-1 (0.00)" {
		t.Fatalf("routing line (no routing) = %q", got.Line)
	}
}

func TestFormatActivityRecoveryAndCollab(t *testing.T) {
	st := collabState(t)

	l1 := FormatActivity(Event{Kind: "recovery.l1_retry", Task: "st-1", Agent: "impl-1", Attempt: 1, ErrorType: "ValueError"}, st)
	if l1.Line != "↻ st-1 · retry #1 (ValueError)" || l1.Class != classErr {
		t.Fatalf("l1 line = %+v", l1)
	}

	guidance := FormatActivity(Event{Kind: "recovery.l2_guidance", Task: "st-1", Agent: "impl-1", Guidance: "add collaborators"}, st)
	if guidance.Line != "mgr-1 guidance → st-1: add collaborators" {
		t.Fatalf("guidance line = %q", guidance.Line)
	}

	reroute := FormatActivity(Event{
		Kind: "recovery.l2_reroute", Task: "st-1", Agent: "impl-1",
		To: "impl-1-collab-1", Guidance: "add collaborators",
	}, st)
	want := "mgr-1 rerouted st-1: impl-1 → impl-1-collab-1 · add collaborators"
	if reroute.Line != want {
		t.Fatalf("reroute line = %q, want %q", reroute.Line, want)
	}

	collab := FormatActivity(Event{Kind: "specialist.collaborator_added", Agent: "impl-1-collab-1", For: "impl-1", Role: "implementer"}, st)
	if collab.Line != "impl-1 + collaborator impl-1-collab-1 (implementer)" {
		t.Fatalf("collab line = %q", collab.Line)
	}
}

func TestFormatActivityGatesTokensRun(t *testing.T) {
	st := collabState(t)

	fail := FormatActivity(Event{
		Kind: "verification.stage", Stage: "3-local-tests", Passed: boolPtr(false),
		Blocking: boolPtr(true), Detail: "2 regressions vs baseline: test_validator.py::test_lazy", DurationSeconds: 19.4,
	}, st)
	if fail.Line != "gate 3-local-tests FAIL (19.4s): 2 regressions vs baseline: test_validator.py::test_lazy" {
		t.Fatalf("gate line = %q", fail.Line)
	}
	if fail.Class != classErr {
		t.Fatalf("gate class = %d", fail.Class)
	}

	skip := FormatActivity(Event{
		Kind: "verification.stage", Stage: "4-code-review", Passed: boolPtr(true),
		Detail: "skipped after blocking failure", DurationSeconds: 0,
	}, st)
	if skip.Line != "gate 4-code-review SKIP (0.0s): skipped after blocking failure" {
		t.Fatalf("skip line = %q", skip.Line)
	}

	usage := FormatActivity(Event{
		Kind: "tokens.usage", Phase: "specialists", GovernorMode: "NORMAL",
		Usage: &UsageBody{PromptTokens: 71000, CompletionTokens: 6100, TotalTokens: 77100},
	}, st)
	if usage.Line != "tokens: 77,100 total (71,000 prompt + 6,100 completion) · phase specialists · NORMAL" {
		t.Fatalf("tokens line = %q", usage.Line)
	}

	end := FormatActivity(Event{Kind: "run.end", Success: boolPtr(true), Outcome: "VERIFIED: all gates passed"}, st)
	if end.Line != "run ended · VERIFIED: all gates passed" || end.Class != classOK {
		t.Fatalf("end line = %+v", end)
	}

	failed := FormatActivity(Event{Kind: "run.failed", Error: "model transport error", Stage: "specialists"}, st)
	if failed.Line != "run FAILED · model transport error" || failed.Class != classErr {
		t.Fatalf("failed line = %+v", failed)
	}
}

func TestFormatActivityAgentPulses(t *testing.T) {
	st := collabState(t)

	step := FormatActivity(Event{
		Kind: "agent.step", Agent: "impl-1", Task: "st-1", Step: 2, MaxSteps: 16, Phase: "thinking",
	}, st)
	if step.Line != "impl-1 · st-1 · step 2/16 · thinking" || step.Class != classDim {
		t.Fatalf("step line = %+v", step)
	}

	usage := FormatActivity(Event{
		Kind: "agent.usage", Agent: "impl-1", Task: "st-1",
		PromptTokens: 15000, CompletionTokens: 1400, TotalTokens: 16400, TotalTokensAgent: 26100,
	}, st)
	if usage.Line != "impl-1: 16,400 tok (call) · 26,100 tok (agent total)" {
		t.Fatalf("usage line = %q", usage.Line)
	}

	result := FormatActivity(Event{
		Kind: "specialist.result", Task: "st-1", Agent: "impl-1-collab-1",
		Success: boolPtr(true), Summary: "validator split into 4 modules; suite green",
	}, st)
	if result.Line != "impl-1-collab-1 → st-1 ✓ validator split into 4 modules; suite green" {
		t.Fatalf("result line = %q", result.Line)
	}
	if result.Class != classOK {
		t.Fatalf("result class = %d", result.Class)
	}
}

func TestFormatActivityPlanAndProfile(t *testing.T) {
	st := collabState(t)

	plan := FormatActivity(Event{
		Kind: "architect.plan", ReproTest: "",
		Subtasks: SubtaskList{{ID: "st-1"}, {ID: "st-2"}},
	}, st)
	if plan.Line != "plan: 2 subtasks · repro no reproduction test" {
		t.Fatalf("plan line = %q", plan.Line)
	}

	profile := FormatActivity(Event{
		Kind: "architect.profile",
		Profile: &ProfileEvent{
			Languages:     map[string]int{"Python": 14},
			TestFramework: "pytest", Files: 14, LOC: 3120,
		},
	}, st)
	if profile.Line != "profile: Python×14 · pytest · 14 files · 3,120 loc" {
		t.Fatalf("profile line = %q", profile.Line)
	}

	start := FormatActivity(Event{Kind: "run.start", RunID: "f7e6d5c4b3a2", Issue: "  Refactor\n  the validator "}, st)
	if start.Line != "run f7e6d5c4 started · Refactor the validator" {
		t.Fatalf("start line = %q", start.Line)
	}

	long := FormatActivity(Event{Kind: "run.start", RunID: "r1234567890", Issue: strings.Repeat("x", 100)}, st)
	if !strings.HasSuffix(long.Line, "…") || !strings.Contains(long.Line, "run ") {
		t.Fatalf("long issue not truncated to 64 chars + ellipsis: %q", long.Line)
	}
	if got := len([]rune(long.Line)); got > len("run ")+8+len(" started · ")+65 {
		t.Fatalf("long start line has %d runes: %q", got, long.Line)
	}
}

func TestFormatActivityBaselineAndUnknown(t *testing.T) {
	st := collabState(t)

	base := FormatActivity(Event{
		Kind: "baseline.captured", Runnable: boolPtr(true), PreExistingFailures: 0, ReproTest: "",
	}, st)
	if base.Line != "baseline: 0 pre-existing failures · repro none" {
		t.Fatalf("baseline line = %q", base.Line)
	}

	notRunnable := FormatActivity(Event{Kind: "baseline.captured", Runnable: boolPtr(false)}, st)
	if notRunnable.Line != "baseline: not runnable" || notRunnable.Class != classErr {
		t.Fatalf("not-runnable = %+v", notRunnable)
	}

	unknown := FormatActivity(Event{Kind: "future.event", RunID: "deadbeef"}, st)
	if unknown.Line != "«future.event» · run deadbeef" || unknown.Class != classDim {
		t.Fatalf("unknown = %+v", unknown)
	}
}

func boolPtr(b bool) *bool { return &b }
