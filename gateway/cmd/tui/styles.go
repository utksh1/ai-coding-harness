// Foreman cockpit TUI — lipgloss theme (slate/blue palette, matching the
// gateway web dashboard: #1e293b / #3b82f6 / #10b981 / #ef4444 / #f59e0b /
// #64748b).
package main

import (
	"strings"

	"github.com/charmbracelet/lipgloss"
)

// Palette.
const (
	colSlate = "#1e293b" // panels
	colBlue  = "#3b82f6" // brand / active / working
	colGreen = "#10b981" // pass / done / connected
	colRed   = "#ef4444" // fail / error
	colAmber = "#f59e0b" // thinking / retry / tokens
	colGray  = "#64748b" // dim / footer
	colText  = "#e2e8f0" // primary text
	colMid   = "#94a3b8" // secondary text
	colEdge  = "#334155" // borders
)

var (
	brandStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("#ffffff")).
			Background(lipgloss.Color(colBlue)).
			Padding(0, 1)

	paneStyle = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color(colEdge)).
			Padding(0, 1)

	paneFocusStyle = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color(colBlue)).
			Padding(0, 1)

	activeTabStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("#ffffff")).
			Background(lipgloss.Color(colBlue)).
			Padding(0, 2)

	inactiveTabStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color(colMid)).
				Background(lipgloss.Color(colSlate)).
				Padding(0, 2)

	footerStyle = lipgloss.NewStyle().Foreground(lipgloss.Color(colGray))

	headerKeyStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color(colGray))
	headerValStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color(colText))
	headerRunStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color("#38bdf8"))
	headerTokenStyle = lipgloss.NewStyle().Foreground(lipgloss.Color(colAmber)).Bold(true)

	okStyle      = lipgloss.NewStyle().Foreground(lipgloss.Color(colGreen)).Bold(true)
	errStyle     = lipgloss.NewStyle().Foreground(lipgloss.Color(colRed)).Bold(true)
	warnStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color(colAmber)).Bold(true)
	blueStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color(colBlue)).Bold(true)
	dimStyle     = lipgloss.NewStyle().Foreground(lipgloss.Color(colGray))
	midStyle     = lipgloss.NewStyle().Foreground(lipgloss.Color(colMid))
	textStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color(colText))
	boldStyle    = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color(colText))
	sectionStyle = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color(colBlue))

	// Org-tree node pieces.
	nodeIDStyle     = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color(colText))
	nodeSelectStyle = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("#ffffff")).
			Background(lipgloss.Color(colSlate))
	nodeRoleStyle     = lipgloss.NewStyle().Foreground(lipgloss.Color(colMid))
	nodeBadgeStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color(colGray))
	nodeActivityStyle = lipgloss.NewStyle().Foreground(lipgloss.Color(colGray))
	treeLineStyle     = lipgloss.NewStyle().Foreground(lipgloss.Color(colGray))
	diffAddStyle      = lipgloss.NewStyle().Foreground(lipgloss.Color(colGreen))
	diffDelStyle      = lipgloss.NewStyle().Foreground(lipgloss.Color(colRed))
	diffHunkStyle     = lipgloss.NewStyle().Foreground(lipgloss.Color("#38bdf8"))

	statusLightStyle = map[AgentStatus]lipgloss.Style{
		StatusIdle:     lipgloss.NewStyle().Foreground(lipgloss.Color(colGray)),
		StatusThinking: lipgloss.NewStyle().Foreground(lipgloss.Color(colAmber)),
		StatusWorking:  lipgloss.NewStyle().Foreground(lipgloss.Color("#38bdf8")),
		StatusDone:     lipgloss.NewStyle().Foreground(lipgloss.Color(colGreen)),
		StatusFailed:   lipgloss.NewStyle().Foreground(lipgloss.Color(colRed)),
	}
)

// statusLight is the per-agent glyph: ● idle / ◐ thinking / ○ working /
// ✓ done / ✗ failed.
func statusLight(st AgentStatus) string {
	switch st {
	case StatusThinking:
		return "◐"
	case StatusWorking:
		return "○"
	case StatusDone:
		return "✓"
	case StatusFailed:
		return "✗"
	default:
		return "●"
	}
}

// taskBadge renders a subtask status badge.
func taskBadge(st TaskStatus) string {
	switch st {
	case TaskRunning:
		return blueStyle.Render("▶ RUNNING")
	case TaskDone:
		return okStyle.Render("✓ DONE   ")
	case TaskFailed:
		return errStyle.Render("✗ FAILED ")
	default:
		return dimStyle.Render("· PENDING")
	}
}

// stageBadge renders a gate verdict badge.
func stageBadge(o StageOutcome) string {
	switch o {
	case StagePass:
		return okStyle.Render("✓ PASS")
	case StageFail:
		return errStyle.Render("✗ FAIL")
	case StageSkip:
		return warnStyle.Render("– SKIP")
	default:
		return dimStyle.Render("○ …   ")
	}
}

// runStateStyle colors the header run state.
func runStateStyle(state string) string {
	switch state {
	case RunVerified:
		return okStyle.Render(state)
	case RunNotVerifed:
		return errStyle.Render(state)
	case RunFailed:
		return errStyle.Render(state)
	case RunRunning:
		return blueStyle.Render(state)
	default:
		return dimStyle.Render("IDLE")
	}
}

// complexityBar renders a 1-10 complexity meter.
func complexityBar(c int) string {
	if c < 0 {
		c = 0
	}
	if c > 10 {
		c = 10
	}
	return strings.Repeat("▰", c) + strings.Repeat("▱", 10-c)
}
