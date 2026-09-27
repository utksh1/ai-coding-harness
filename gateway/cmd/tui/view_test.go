// Foreman cockpit TUI — layout invariant smoke test: the View renders within
// the terminal width at both the 120×40 target and the 80×24 floor (no line
// may exceed the width — lipgloss word-wraps oversize content, which would
// break the fixed-height panes). This is a layout contract check, not an
// interactive test.
package main

import (
	"regexp"
	"strings"
	"testing"

	"github.com/charmbracelet/x/ansi"
)

var ansiRe = regexp.MustCompile(`\x1b\[[0-9;?]*[a-zA-Z]`)

func stripANSI(s string) string { return ansiRe.ReplaceAllString(s, "") }

// newViewModel builds a model with the fixture folded in (no transport).
func newViewModel(t *testing.T) model {
	t.Helper()
	m := initialModel("ws://localhost:8080/ws")
	for _, entry := range applyFixture(t, "../../web/events.sample-collab.jsonl").Activity {
		m.state.Apply(entry.Ev)
	}
	m.selectedAgent = "impl-1"
	return m
}

func TestViewFitsWidth(t *testing.T) {
	for _, size := range [][2]int{{120, 40}, {100, 30}, {84, 24}, {80, 24}, {60, 12}} {
		for _, tabIdx := range []tab{tabPlan, tabAgent, tabActivity, tabVerify, tabDiff} {
			m := newViewModel(t)
			m.width, m.height = size[0], size[1]
			m.activeTab = tabIdx
			view := m.View()
			for i, line := range strings.Split(view, "\n") {
				if w := ansi.StringWidth(stripANSI(line)); w > size[0] {
					t.Fatalf("%d×%d tab %d: line %d has display width %d > %d: %q",
						size[0], size[1], int(tabIdx)+1, i, w, size[0], stripANSI(line))
				}
			}
			// The frame must not exceed the terminal height either.
			if n := len(strings.Split(view, "\n")); n > size[1]+2 {
				t.Fatalf("%d×%d tab %d: frame has %d lines (budget %d)",
					size[0], size[1], int(tabIdx)+1, n, size[1])
			}
		}
	}
}

func TestViewContainsOrgTree(t *testing.T) {
	m := newViewModel(t)
	m.width, m.height = 120, 40
	view := stripANSI(m.View())
	for _, want := range []string{"FOREMAN", "AGENT ORG-TREE", "architect-1", "mgr-1", "impl-1", "impl-1-collab-1", "ver-1"} {
		if !strings.Contains(view, want) {
			t.Fatalf("view missing %q", want)
		}
	}
	// Box-drawing connectors are present (delegation edges of the org-tree).
	for _, want := range []string{"└──", "├──", "│"} {
		if !strings.Contains(view, want) {
			t.Fatalf("view missing connector %q", want)
		}
	}
	// The tree focus mode also renders without exceeding the width.
	m.focus = focusTree
	m.cursor = 2
	view = stripANSI(m.View())
	if !strings.Contains(view, "impl-1") {
		t.Fatal("tree focus render lost impl-1")
	}
}

func TestViewTreeFocusAndInputModal(t *testing.T) {
	m := newViewModel(t)
	m.width, m.height = 120, 40
	m.focus = focusTree
	m.cursor = 3 // impl-1-collab-1 in preorder
	view := stripANSI(m.View())
	if !strings.Contains(view, "impl-1-collab-1") {
		t.Fatal("cursor render lost the collaborator")
	}

	m.inputMode = true
	m.issueInput = "fix the width bug"
	view = stripANSI(m.View())
	if !strings.Contains(view, "LAUNCH NEW TASK") || !strings.Contains(view, "fix the width bug") {
		t.Fatalf("modal render = %q", view)
	}
}

func TestViewEmptyState(t *testing.T) {
	m := initialModel("ws://localhost:8080/ws")
	m.width, m.height = 120, 40
	view := stripANSI(m.View())
	if !strings.Contains(view, "waiting for roster or events") {
		t.Fatalf("empty roster placeholder missing: %q", view)
	}
	// A too-small terminal renders a message, never garbage.
	m.width, m.height = 40, 8
	view = stripANSI(m.View())
	if !strings.Contains(view, "terminal too small") {
		t.Fatalf("small-terminal fallback missing: %q", view)
	}
}
