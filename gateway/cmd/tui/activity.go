// Foreman cockpit TUI — human-readable activity lines (pure).
//
// FormatActivity turns one event into a single compact feed line, e.g.
//
//	impl-1 → st-1: read parse/core.py (12ms)
//	mgr-1 routed st-1 → impl-1 (0.80)
//
// The state parameter supplies context the event itself lacks (the manager
// id, roster facts). Unknown kinds fall back to the generic contract line.
package main

import (
	"fmt"
	"strings"
)

// lineClass drives the feed's coloring.
type lineClass int

const (
	classNormal lineClass = iota
	classDim
	classOK
	classWarn
	classErr
)

// formattedLine is one rendered feed entry.
type formattedLine struct {
	Line  string
	Class lineClass
}

// FormatActivity renders one event as a human-readable feed line.
func FormatActivity(ev Event, s *State) formattedLine {
	switch ev.Kind {
	case "run.start":
		issue := oneLine(ev.Issue)
		if len(issue) > 64 {
			issue = issue[:64] + "…"
		}
		if issue == "" {
			return formattedLine{"run " + shortRun(ev.RunID) + " started", classOK}
		}
		return formattedLine{"run " + shortRun(ev.RunID) + " started · " + issue, classOK}

	case "run.end":
		if ev.Success != nil && *ev.Success {
			return formattedLine{"run ended · " + ev.Outcome, classOK}
		}
		detail := ev.Outcome
		if ev.StopReason != "" {
			detail = firstNonEmpty(ev.Outcome, "stopped") + " · " + ev.StopReason
		}
		return formattedLine{"run ended · " + detail, classErr}

	case "run.failed":
		return formattedLine{"run FAILED · " + firstNonEmpty(ev.Error, ev.Stage), classErr}

	case "budget.exhausted":
		reason := ev.Reason
		if reason == "" {
			reason = "budget governor tripped"
		}
		return formattedLine{"⏱ run stopped · " + reason, classErr}

	case "architect.profile":
		return formattedLine{"profile: " + profileLine(ev.Profile), classDim}

	case "architect.retry":
		detail := ev.Error
		if detail == "" {
			detail = "model unavailable"
		}
		return formattedLine{
			fmt.Sprintf("architect attempt %d failed · retrying · %s", ev.Attempt, detail),
			classWarn,
		}

	case "architect.plan":
		repro := ev.ReproTest
		if repro == "" {
			repro = "no reproduction test"
		}
		return formattedLine{
			fmt.Sprintf("plan: %d subtasks · repro %s", len(ev.Subtasks), repro),
			classNormal,
		}

	case "baseline.captured":
		if ev.Runnable != nil && !*ev.Runnable {
			return formattedLine{"baseline: not runnable", classErr}
		}
		repro := ev.ReproTest
		if repro == "" {
			repro = "none"
		}
		return formattedLine{
			fmt.Sprintf("baseline: %d pre-existing failures · repro %s", ev.PreExistingFailures, repro),
			classDim,
		}

	case "specialist.assigned":
		mgr := firstNonEmpty(s.ManagerID, "mgr")
		return formattedLine{
			fmt.Sprintf("%s routed %s → %s (%s)", mgr, ev.Task, ev.Agent, scoreFmt(routingTotalOf(ev))),
			classNormal,
		}

	case "specialist.result":
		status := "✓"
		cls := classOK
		if ev.Success != nil && !*ev.Success {
			status = "✗"
			cls = classErr
		}
		return formattedLine{
			fmt.Sprintf("%s → %s %s %s", ev.Agent, ev.Task, status, ev.Summary),
			cls,
		}

	case "specialist.collaborator_added":
		return formattedLine{
			fmt.Sprintf("%s + collaborator %s (%s)", ev.For, ev.Agent, ev.Role),
			classNormal,
		}

	case "agent.step":
		return formattedLine{
			fmt.Sprintf("%s · %s · step %d/%d · %s", ev.Agent, ev.Task, ev.Step, ev.MaxSteps, ev.Phase),
			classDim,
		}

	case "agent.tool":
		ok := ev.OK == nil || *ev.OK
		line := fmt.Sprintf("%s → %s: %s %s (%dms)",
			ev.Agent, ev.Task, toolVerb(ev.Tool), ev.ArgsDigest, ev.DurationMS)
		if !ok {
			line += " ✗ " + ev.ResultDigest
			return formattedLine{line, classErr}
		}
		return formattedLine{line, classNormal}

	case "agent.usage":
		return formattedLine{
			fmt.Sprintf("%s: %s tok (call) · %s tok (agent total)",
				ev.Agent, thousands(ev.TotalTokens), thousands(ev.TotalTokensAgent)),
			classDim,
		}

	case "recovery.l1_retry":
		return formattedLine{
			fmt.Sprintf("↻ %s · retry #%d (%s)", ev.Task, ev.Attempt, ev.ErrorType),
			classErr,
		}

	case "recovery.l2_guidance":
		return formattedLine{
			fmt.Sprintf("%s guidance → %s: %s", firstNonEmpty(s.ManagerID, "mgr"), ev.Task, ev.Guidance),
			classErr,
		}

	case "recovery.l2_reroute":
		return formattedLine{
			fmt.Sprintf("%s rerouted %s: %s → %s · %s",
				firstNonEmpty(s.ManagerID, "mgr"), ev.Task, ev.Agent, ev.To, ev.Guidance),
			classErr,
		}

	case "recovery.l3_replan":
		return formattedLine{ev.Task + " · L3 replan (architect re-plans)", classErr}

	case "recovery.l3_skipped_budget":
		return formattedLine{ev.Task + " · L3 skipped: token budget exhausted", classErr}

	case "verification.stage":
		verdict := "—"
		cls := classNormal
		if ev.Passed != nil {
			if *ev.Passed {
				verdict, cls = "PASS", classOK
			} else {
				verdict, cls = "FAIL", classErr
			}
		}
		if ev.Skipped != nil && *ev.Skipped || startsWithFold(ev.Detail, "skipped") {
			verdict, cls = "SKIP", classDim
		}
		return formattedLine{
			fmt.Sprintf("gate %s %s (%.1fs): %s", ev.Stage, verdict, ev.DurationSeconds, ev.Detail),
			cls,
		}

	case "tokens.usage":
		usage := ev.Usage
		total, prompt, completion := 0, 0, 0
		if usage != nil {
			total, prompt, completion = usage.TotalTokens, usage.PromptTokens, usage.CompletionTokens
		}
		return formattedLine{
			fmt.Sprintf("tokens: %s total (%s prompt + %s completion) · phase %s · %s",
				thousands(total), thousands(prompt), thousands(completion), ev.Phase, ev.GovernorMode),
			classDim,
		}

	case "task.completed":
		return formattedLine{"gateway: task.completed (final response recorded)", classDim}

	default:
		// Unknown kind: generic kind + run_id, never dropped.
		return formattedLine{fmt.Sprintf("«%s» · run %s", ev.Kind, shortRun(ev.RunID)), classDim}
	}
}

// toolVerb maps tool names to feed verbs ("read_file" → "read").
func toolVerb(tool string) string {
	switch tool {
	case "read_file":
		return "read"
	case "write_file":
		return "write"
	case "apply_edit":
		return "edit"
	case "run_tests":
		return "run tests"
	case "grep_search":
		return "grep"
	case "glob_search":
		return "glob"
	case "list_dir":
		return "ls"
	case "bash", "run_command":
		return "exec"
	case "":
		return "tool"
	default:
		return strings.ReplaceAll(tool, "_", " ")
	}
}

// profileLine summarizes architect.profile for the feed/plan header.
func profileLine(p *ProfileEvent) string {
	if p == nil {
		return "unknown"
	}
	var parts []string
	if n := len(p.Languages); n > 0 {
		top := ""
		topCount := 0
		for lang, count := range p.Languages {
			if count > topCount {
				top, topCount = lang, count
			}
		}
		if n == 1 {
			parts = append(parts, fmt.Sprintf("%s×%d", top, topCount))
		} else {
			parts = append(parts, fmt.Sprintf("%s×%d (+%d more)", top, topCount, n-1))
		}
	}
	if p.TestFramework != "" {
		parts = append(parts, p.TestFramework)
	}
	if p.Files > 0 {
		parts = append(parts, fmt.Sprintf("%d files", p.Files))
	}
	if p.LOC > 0 {
		parts = append(parts, thousands(p.LOC)+" loc")
	}
	if len(parts) == 0 {
		return "unknown"
	}
	return strings.Join(parts, " · ")
}

// routingTotalOf extracts the routing score total (0 when absent).
func routingTotalOf(ev Event) float64 {
	if ev.Routing == nil {
		return 0
	}
	return ev.Routing.Total
}

// scoreFmt renders a routing score as "0.80".
func scoreFmt(f float64) string {
	return fmt.Sprintf("%.2f", f)
}

// thousands renders 114000 → "114,000".
func thousands(n int) string {
	neg := n < 0
	if neg {
		n = -n
	}
	s := fmt.Sprintf("%d", n)
	var b strings.Builder
	for i, r := range s {
		if i > 0 && (len(s)-i)%3 == 0 {
			b.WriteByte(',')
		}
		b.WriteRune(r)
	}
	if neg {
		return "-" + b.String()
	}
	return b.String()
}

// fmtTokens renders a compact token counter: 45210 → "45.2k", 940 → "940".
func fmtTokens(n int) string {
	switch {
	case n >= 1_000_000:
		return fmt.Sprintf("%.1fM", float64(n)/1_000_000)
	case n >= 10_000:
		return fmt.Sprintf("%.0fk", float64(n)/1000)
	case n >= 1000:
		return fmt.Sprintf("%.1fk", float64(n)/1000)
	default:
		return fmt.Sprintf("%d", n)
	}
}

// shortRun truncates a run id to its first 8 chars.
func shortRun(runID string) string {
	if len(runID) <= 8 {
		return runID
	}
	return runID[:8]
}

// oneLine collapses whitespace (events may carry multi-line issues).
func oneLine(s string) string {
	return strings.TrimSpace(strings.Join(strings.Fields(s), " "))
}

func firstNonEmpty(xs ...string) string {
	for _, x := range xs {
		if x != "" {
			return x
		}
	}
	return ""
}
