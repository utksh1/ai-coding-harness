// Foreman cockpit TUI — rendering: header, the org-tree pane, the five main
// tabs, the new-task modal and the footer. Compact 120×40 target, graceful
// degradation to 80×24 (skills badges hidden and lines shortened below
// width 100).
package main

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/x/ansi"
)

// View renders the whole cockpit frame.
func (m model) View() string {
	if m.width < 56 || m.height < 10 {
		return dimStyle.Render(fmt.Sprintf(
			" terminal too small: %d×%d — need at least 56×10", m.width, m.height))
	}
	var b strings.Builder
	b.WriteString(m.renderHeader())
	b.WriteString("\n")
	b.WriteString(m.renderBody())
	b.WriteString("\n")
	b.WriteString(m.renderFooter())
	return b.String()
}

// renderHeader: brand · connection · run · tokens · governor · verdict.
func (m model) renderHeader() string {
	s := m.state
	narrow := m.width < 100

	var parts []string
	parts = append(parts, brandStyle.Render("FOREMAN"))

	switch {
	case m.replay != nil:
		glyph, style := "▶", warnStyle
		if m.replay.paused {
			glyph, style = "⏸", midStyle
		}
		parts = append(parts, style.Render(fmt.Sprintf("%s REPLAY %d/%d",
			glyph, m.replay.idx, m.replay.total())))
		if !narrow && m.replay.speed > 0 {
			parts = append(parts, dimStyle.Render(fmt.Sprintf("%dms/event", m.replay.speed.Milliseconds())))
		}
	case m.connected:
		parts = append(parts, okStyle.Render("● LIVE"))
	default:
		parts = append(parts, errStyle.Render("○ OFFLINE"))
	}

	if s.RunID != "" {
		parts = append(parts, headerKeyStyle.Render("run"), headerRunStyle.Render(shortRun(s.RunID)))
	}
	if s.TotalTokens > 0 || s.RunState != "" {
		parts = append(parts, headerTokenStyle.Render("◆ "+thousands(s.TotalTokens)+" tok"))
	}
	if s.GovernorMode != "" && !narrow {
		parts = append(parts, dimStyle.Render(s.GovernorMode))
	}
	if s.RunState != "" {
		parts = append(parts, runStateStyle(s.RunState))
	}
	return truncWidth(strings.Join(parts, "  "), m.width)
}

// renderBody: the org-tree pane (left, always visible) + main pane.
func (m model) renderBody() string {
	narrow := m.width < 100
	treeW := 40
	switch {
	case m.width < 80:
		treeW = 28
	case narrow:
		treeW = 32
	}
	bodyH := m.height - 2 // header + footer

	treeStyle := paneStyle
	if m.focus == focusTree {
		treeStyle = paneFocusStyle
	}
	treeBox := treeStyle.
		Height(bodyH - 2).
		Width(treeW - 4).
		Render(m.renderTreeContent(treeW-6, bodyH-4, narrow))

	mainBox := paneStyle.
		Height(bodyH - 2).
		Width(m.width - treeW - 5).
		Render(m.renderMainContent(m.width-treeW-7, bodyH-4))

	return lipgloss.JoinHorizontal(lipgloss.Top, treeBox, " ", mainBox)
}

// ---------------------------------------------------------------------------
// Org-tree pane

// renderTreeContent renders the agent org-tree with box connectors.
func (m model) renderTreeContent(innerW, innerH int, narrow bool) string {
	s := m.state
	var lines []string

	title := sectionStyle.Render("AGENT ORG-TREE") + " " +
		dimStyle.Render("("+s.RosterSource+")")
	lines = append(lines, truncWidth(title, innerW))

	if s.Root == nil {
		lines = append(lines, "", dimStyle.Render(" waiting for roster or events…"))
		return padLines(lines, innerH)
	}

	compact := m.height < 24
	flat := s.FlatTree()
	if m.cursor >= len(flat) {
		m.cursor = len(flat) - 1
	}

	lines = append(lines, m.renderNode(s.Root, "", "", innerW, narrow, compact)...)

	// Window to fit: keep the head (title + root) and as much tree as fits.
	if len(lines) > innerH {
		hidden := len(lines) - innerH + 1
		lines = append(lines[:innerH-1], dimStyle.Render(
			fmt.Sprintf(" … +%d more rows", hidden)))
	}
	return padLines(lines, innerH)
}

// renderNode renders one org-tree node and its subtree.
//
// headPrefix is the (plain) indent in front of this node's status glyph —
// for children it ends in the box connector ("├── " / "└── "). rail is the
// continuation prefix used both for this node's own extra lines and for
// its children ("│   " while later siblings follow, "    " when last).
func (m model) renderNode(n *AgentNode, headPrefix, rail string, innerW int, narrow, compact bool) []string {
	if n == nil {
		return nil
	}
	extra := rail
	if extra == "" {
		extra = "  " // root node extra lines get a small indent
	}

	out := []string{m.treeHeadLine(n, headPrefix, innerW)}
	if !compact {
		out = append(out, m.treeRoleLine(n, extra, innerW, narrow))
	}
	if act := agentActivityText(n); act != "" {
		out = append(out, truncWidth(extra+nodeActivityStyle.Render(act), innerW))
	}

	last := len(n.Children) - 1
	for i, c := range n.Children {
		conn, crail := "├── ", rail+"│   "
		if i == last {
			conn, crail = "└── ", rail+"    "
		}
		out = append(out, m.renderNode(c, rail+conn, crail, innerW, narrow, compact)...)
	}
	return out
}

// treeHeadLine is "… └── ○ impl-1 ······ 46.9k" with the selection bar.
func (m model) treeHeadLine(n *AgentNode, headPrefix string, innerW int) string {
	tok := fmtTokens(n.Tokens)

	flat := m.state.FlatTree()
	selected := m.focus == focusTree && m.cursor < len(flat) && flat[m.cursor] == n

	idText := n.ID
	if n.Retries > 0 {
		idText += fmt.Sprintf(" ↻%d", n.Retries)
	}
	light := statusLight(n.Status)
	used := lipgloss.Width(headPrefix) + lipgloss.Width(light) + 1 +
		lipgloss.Width(idText) + 1 + lipgloss.Width(tok)
	fill := innerW - used

	var body string
	if fill >= 1 {
		body = light + " " + idText + strings.Repeat(" ", fill) + tok
	} else {
		body = light + " " + idText + " " + tok
	}
	body = truncWidth(body, innerW-lipgloss.Width(headPrefix))

	if selected {
		return headPrefix + nodeSelectStyle.Render(body)
	}
	return headPrefix + statusLightStyle[n.Status].Render(light) + " " +
		nodeIDStyle.Render(idText) + " " + dimStyle.Render(tok)
}

// treeRoleLine is the role + skills badges + delegation decoration.
func (m model) treeRoleLine(n *AgentNode, extra string, innerW int, narrow bool) string {
	role := n.Role
	if role == "" {
		role = "?"
	}
	text := role
	if n.ToolTier > 0 && !narrow {
		text += " · T" + fmt.Sprint(n.ToolTier)
	}
	if !narrow && len(n.Specialties) > 0 {
		badges := strings.Join(n.Specialties, ",")
		if len(badges) > 18 {
			badges = badges[:18] + "…"
		}
		text += " · " + badges
	}
	if n.AssignedTask != "" {
		text += fmt.Sprintf(" · ⟵ %s", n.AssignedTask)
		if n.RoutingTotal > 0 {
			text += " (" + scoreFmt(n.RoutingTotal) + ")"
		}
	}
	return truncWidth(extra+nodeRoleStyle.Render(text), innerW)
}

// agentActivityText is the per-agent "what are they doing" pulse:
// "st-1 · step 3/16 · read parse/core.py".
func agentActivityText(n *AgentNode) string {
	var parts []string
	if t := firstNonEmpty(n.AssignedTask, n.Task); t != "" && n.Level >= 2 {
		parts = append(parts, t)
	}
	if n.MaxSteps > 0 {
		parts = append(parts, fmt.Sprintf("step %d/%d", n.Step, n.MaxSteps))
	}
	if n.LastTool != "" {
		parts = append(parts, toolVerb(n.LastTool)+" "+n.LastArgs)
	} else if n.Phase != "" {
		parts = append(parts, n.Phase)
	}
	if len(parts) == 0 {
		if n.Status == StatusFailed {
			return "failed"
		}
		if n.Summary != "" {
			return oneLine(n.Summary)
		}
		return "idle"
	}
	return strings.Join(parts, " · ")
}

// ---------------------------------------------------------------------------
// Main pane

// renderMainContent: tab bar + active tab body.
func (m model) renderMainContent(innerW, innerH int) string {
	var lines []string
	lines = append(lines, m.renderTabBar(innerW))
	lines = append(lines, m.renderTabDivider(innerW))

	var body []string
	if m.inputMode {
		body = m.renderInputModal(innerW, innerH-2)
	} else {
		switch m.activeTab {
		case tabRuns:
			body = m.renderRunsTab(innerW, innerH-4)
		case tabPlan:
			body = m.renderPlanTab(innerW)
		case tabAgent:
			body = m.renderAgentTab(innerW)
		case tabActivity:
			body = m.renderActivityTab(innerW)
		case tabVerify:
			body = m.renderVerifyTab(innerW)
		case tabDiff:
			body = m.renderDiffTab(innerW)
		}
	}
	body = windowLines(body, innerH-2)
	lines = append(lines, body...)
	return padLines(lines, innerH)
}

// renderTabBar: [1 Plan] [2 Agent] … + right-side notice.
func (m model) renderTabBar(innerW int) string {
	s := m.state
	counts := []string{
		fmt.Sprint(len(m.runs)),
		fmt.Sprint(len(s.TaskOrder())),
		"",
		fmt.Sprint(len(s.Activity)),
		fmt.Sprintf("%d/%d", stagePassed(s), len(s.StageOrder())),
		"",
	}
	var rendered []string
	for i, name := range tabNames {
		label := fmt.Sprintf("%d %s", i+1, name)
		if counts[i] != "" && lipgloss.Width(label)+len(counts[i])+3 <= 14 {
			label += "(" + counts[i] + ")"
		}
		if tab(i) == m.activeTab {
			rendered = append(rendered, activeTabStyle.Render(label))
		} else {
			rendered = append(rendered, inactiveTabStyle.Render(label))
		}
	}
	left := strings.Join(rendered, " ")
	right := ""
	if m.inputMode {
		right = warnStyle.Render("NEW TASK")
	} else if m.notice != "" {
		right = midStyle.Render(oneLine(m.notice))
	} else if !m.connected && m.replay == nil && m.connDetail != "" {
		right = midStyle.Render(oneLine(m.connDetail))
	}
	avail := innerW - lipgloss.Width(left) - 2
	if right != "" && avail > 4 {
		return left + "  " + truncWidth(right, avail)
	}
	return truncWidth(left, innerW)
}

func (m model) renderTabDivider(innerW int) string {
	return dimStyle.Render(strings.Repeat("─", max(0, innerW)))
}

// stagePassed counts PASS gates.
func stagePassed(s *State) int {
	n := 0
	for _, st := range s.Stages {
		if st.Outcome == StagePass {
			n++
		}
	}
	return n
}

// ---------------------------------------------------------------------------
// Tab 1 — Plan: the subtask board

func (m model) renderPlanTab(innerW int) []string {
	s := m.state
	var lines []string

	var head []string
	if s.Profile != nil {
		head = append(head, "repo: "+profileLine(s.Profile))
	}
	if s.ReproTest != "" {
		head = append(head, "repro: "+s.ReproTest)
	}
	if len(head) > 0 {
		lines = append(lines, truncWidth(midStyle.Render(" · "+strings.Join(head, " · ")), innerW))
	}

	ids := s.TaskOrder()
	if len(ids) == 0 {
		lines = append(lines, "", dimStyle.Render(" no plan yet — subtasks appear on architect.plan"))
		return lines
	}
	if s.BaselineFailures > 0 || s.ReproTest != "" {
		base := fmt.Sprintf("baseline: %d pre-existing failures", s.BaselineFailures)
		lines = append(lines, truncWidth(dimStyle.Render(" "+base), innerW))
	}
	lines = append(lines, "")

	narrow := m.width < 100
	for _, id := range ids {
		t := s.Tasks[id]
		if t == nil {
			continue
		}
		lines = append(lines, m.renderTaskCard(t, innerW, narrow)...)
		lines = append(lines, "")
	}
	return lines
}

func (m model) renderTaskCard(t *Task, innerW int, narrow bool) []string {
	var lines []string

	title := firstNonEmpty(t.Title, "—")
	head := fmt.Sprintf("%s %s", taskBadge(t.Status), boldStyle.Render(t.ID+" · "+title))
	if t.Retries > 0 {
		head += " " + warnStyle.Render(fmt.Sprintf("↻%d", t.Retries))
	}
	if t.ReroutedTo != "" {
		head += " " + warnStyle.Render("⇄ "+t.ReroutedTo)
	}
	lines = append(lines, truncWidth(head, innerW))

	meta := firstNonEmpty(t.Agent, "unassigned")
	if t.CompletedBy != "" && t.CompletedBy != t.Agent {
		meta += " → " + t.CompletedBy
	}
	if t.Specialty != "" {
		meta += " · " + t.Specialty
	}
	if t.Complexity > 0 {
		meta += fmt.Sprintf(" · %s %d", complexityBar(t.Complexity), t.Complexity)
	}
	if !narrow && len(t.Files) > 0 {
		meta += " · " + strings.Join(t.Files, ",")
	}
	if len(t.DependsOn) > 0 {
		meta += " · after " + strings.Join(t.DependsOn, ",")
	}
	if t.Batch > 0 {
		meta += fmt.Sprintf(" · batch %d", t.Batch)
	}
	if t.RoutingTotal > 0 {
		meta += " · score " + scoreFmt(t.RoutingTotal)
	}
	if t.RoutingDetail != "" && !narrow {
		meta += dimStyle.Render(" (" + t.RoutingDetail + ")")
	}
	lines = append(lines, truncWidth("   "+midStyle.Render(meta), innerW))

	if t.Summary != "" {
		mark := okStyle.Render("✓ ")
		if t.Status == TaskFailed {
			mark = errStyle.Render("✗ ")
		}
		lines = append(lines, truncWidth("   "+mark+dimStyle.Render(oneLine(t.Summary)), innerW))
	}
	if t.Acceptance != "" && !narrow {
		lines = append(lines, truncWidth("   "+dimStyle.Render("accept: "+oneLine(t.Acceptance)), innerW))
	}
	return lines
}

// ---------------------------------------------------------------------------
// Tab 2 — Agent: selected agent detail + full tool-call log

func (m model) renderAgentTab(innerW int) []string {
	s := m.state
	if m.selectedAgent == "" {
		return []string{"", dimStyle.Render(" select an agent from the tree: [t] focus · [↑/↓] move · [enter] open")}
	}
	n := s.Agent(m.selectedAgent)
	if n == nil {
		return []string{"", dimStyle.Render(" agent " + m.selectedAgent + " unknown")}
	}
	var lines []string

	head := statusLightStyle[n.Status].Render(statusLight(n.Status)) + " " +
		boldStyle.Render(n.ID) + " · " + midStyle.Render(firstNonEmpty(n.Role, "?"))
	if n.Level > 0 {
		head += midStyle.Render(fmt.Sprintf(" · L%d", n.Level))
	}
	head += " · " + headerTokenStyle.Render(thousands(n.Tokens)+" tok")
	if n.MaxSteps > 0 {
		head += midStyle.Render(fmt.Sprintf(" · step %d/%d", n.Step, n.MaxSteps))
	}
	lines = append(lines, truncWidth(head, innerW))

	sub := firstNonEmpty(strings.Join(n.Specialties, ", "), "specialties unknown")
	if n.ToolTier > 0 {
		sub += fmt.Sprintf(" · tool tier %d", n.ToolTier)
	}
	if n.Model != "" && m.width >= 100 {
		sub += " · model " + n.Model
	}
	lines = append(lines, truncWidth("   "+dimStyle.Render(sub), innerW))

	assign := "task: " + firstNonEmpty(n.Task, "—")
	if n.AssignedTask != "" {
		assign = "task: " + n.AssignedTask
		if n.RoutingTotal > 0 {
			assign += " · routed " + scoreFmt(n.RoutingTotal)
		}
		if n.RoutingDetail != "" {
			assign += " (" + n.RoutingDetail + ")"
		}
	}
	if n.Retries > 0 {
		assign += warnStyle.Render(fmt.Sprintf(" · retries %d", n.Retries))
	}
	if n.ReroutedTo != "" {
		assign += warnStyle.Render(" · rerouted to " + n.ReroutedTo)
	}
	lines = append(lines, truncWidth("   "+midStyle.Render(assign), innerW))
	if n.Summary != "" {
		lines = append(lines, truncWidth("   "+dimStyle.Render("last: "+oneLine(n.Summary)), innerW))
	}
	lines = append(lines, "",
		truncWidth(sectionStyle.Render(fmt.Sprintf("TOOL CALLS (%d)", len(n.ToolCalls))), innerW))

	if len(n.ToolCalls) == 0 {
		lines = append(lines, dimStyle.Render(" no tool calls recorded yet"))
		return lines
	}

	// Tail-scrolled table.
	rows := make([]string, len(n.ToolCalls))
	for i, tc := range n.ToolCalls {
		rows[i] = renderToolRow(tc, innerW, m.width >= 100)
	}
	visible := windowTail(rows, m.agentLogOffset, max(1, tabBodyHeight(m)-len(lines)-1))
	lines = append(lines, visible...)
	return lines
}

func renderToolRow(tc ToolCall, innerW int, wide bool) string {
	okMark := okStyle.Render("✓")
	if !tc.OK {
		okMark = errStyle.Render("✗")
	}
	step := fmt.Sprintf("%2d", tc.Step)
	if tc.Step == 0 {
		step = " ·"
	}
	dur := fmt.Sprintf("%4dms", tc.DurationMS)
	args := oneLine(tc.ArgsDigest)
	tail := " " + okMark
	if wide && tc.Result != "" {
		tail += " " + oneLine(tc.Result)
	}
	head := fmt.Sprintf(" %s %-12s ", step, tc.Tool)
	// Reserve room for the duration + tail before truncating the args.
	argsW := innerW - lipgloss.Width(head) - len(dur) - 1 - lipgloss.Width(tail)
	if argsW < 1 {
		argsW = 1
	}
	line := head + truncWidth(args, argsW) + " " + dimStyle.Render(dur) + tail
	return truncWidth(line, innerW)
}

// ---------------------------------------------------------------------------
// Tab 3 — Activity: chronological human-readable feed (last 200 rendered)

const activityRenderCap = 200

func (m model) renderActivityTab(innerW int) []string {
	s := m.state
	entries := s.Activity
	if len(entries) == 0 {
		return []string{"", dimStyle.Render(" no events yet — waiting for the engine stream…")}
	}
	start := 0
	if len(entries) > activityRenderCap {
		start = len(entries) - activityRenderCap
	}
	entries = entries[start:]

	rows := make([]string, len(entries))
	for i, e := range entries {
		fl := FormatActivity(e.Ev, s)
		style := textStyle
		switch fl.Class {
		case classDim:
			style = dimStyle
		case classOK:
			style = okStyle
		case classWarn:
			style = warnStyle
		case classErr:
			style = errStyle
		}
		rows[i] = dimStyle.Render(fmt.Sprintf(" #%3d ", e.Seq)) + style.Render(fl.Line)
	}
	// Every row is hard-truncated to the pane width (lipgloss Width wraps,
	// which would break the fixed-height pane).
	for i := range rows {
		rows[i] = truncWidth(rows[i], innerW)
	}
	visible := windowTail(rows, m.activityOffset, max(1, tabBodyHeight(m)))
	if start > 0 {
		visible = append([]string{dimStyle.Render(fmt.Sprintf(" … %d earlier events hidden", start))}, visible...)
	}
	return visible
}

// ---------------------------------------------------------------------------
// Tab 4 — Verify: six gates + verdict

func (m model) renderVerifyTab(innerW int) []string {
	s := m.state
	var lines []string

	if s.RunState != "" {
		verdict := "verdict: " + s.Outcome
		switch s.RunState {
		case RunVerified:
			lines = append(lines, truncWidth(okStyle.Render("✓ "+verdict), innerW))
		case RunNotVerifed, RunFailed:
			lines = append(lines, truncWidth(errStyle.Render("✗ "+verdict), innerW))
		default:
			lines = append(lines, truncWidth(blueStyle.Render("◆ "+verdict), innerW))
		}
	} else {
		lines = append(lines, truncWidth(dimStyle.Render("verdict: run in progress…"), innerW))
	}
	lines = append(lines, "")

	order := s.StageOrder()
	if len(order) == 0 {
		lines = append(lines, dimStyle.Render(" no verification gates yet — they run after the specialists finish"))
		return lines
	}
	for _, name := range order {
		st := s.Stages[name]
		if st == nil {
			continue
		}
		blocking := ""
		if st.Blocking {
			blocking = dimStyle.Render(" !")
		}
		dur := ""
		if st.Seconds > 0 {
			dur = fmt.Sprintf("%5.1fs", st.Seconds)
		} else {
			dur = "    —  "
		}
		line := " " + stageBadge(st.Outcome) + blocking + " " + midStyle.Render(padRight(name, 14)) +
			" " + dimStyle.Render(dur) + " " + textStyle.Render(oneLine(st.Detail))
		lines = append(lines, truncWidth(line, innerW))
	}
	return lines
}

// ---------------------------------------------------------------------------
// Tab 5 — Diff: patch.diff evidence, +/- colored

func (m model) renderDiffTab(innerW int) []string {
	var lines []string
	switch m.diffStatus {
	case "":
		lines = append(lines, dimStyle.Render(" the patch loads here when the run ends (evidence patch.diff)"))
		if m.replay != nil {
			lines = append(lines, dimStyle.Render(" replay mode: no evidence endpoint is contacted"))
		}
		return lines
	case "fetching":
		return []string{"", dimStyle.Render(" fetching patch.diff…")}
	case "none":
		return []string{"", warnStyle.Render(" no patch on record for this run")}
	case "error":
		return []string{"", errStyle.Render(" evidence fetch failed: " + oneLine(m.diffErr))}
	}

	if len(m.diffLines) == 0 {
		return []string{"", dimStyle.Render(" empty patch — no changes recorded")}
	}

	added, deleted := 0, 0
	for _, l := range m.diffLines {
		switch {
		case strings.HasPrefix(l, "+"):
			added++
		case strings.HasPrefix(l, "-"):
			deleted++
		}
	}
	header := fmt.Sprintf(" patch.diff · %s +%d / -%d", m.diffFetchedRun, added, deleted)
	lines = append(lines, truncWidth(midStyle.Render(header), innerW), "")

	rendered := make([]string, len(m.diffLines))
	for i, l := range m.diffLines {
		switch {
		case strings.HasPrefix(l, "+"):
			rendered[i] = diffAddStyle.Render(l)
		case strings.HasPrefix(l, "-"):
			rendered[i] = diffDelStyle.Render(l)
		case strings.HasPrefix(l, "@@"):
			rendered[i] = diffHunkStyle.Render(l)
		default:
			rendered[i] = midStyle.Render(l)
		}
		// Diffs carry unbounded line lengths: hard-truncate to the pane width
		// (lipgloss Width would word-wrap and break the fixed-height pane).
		rendered[i] = truncWidth(rendered[i], innerW)
	}
	visible := windowHead(rendered, m.diffOffset, max(1, tabBodyHeight(m)-2))
	lines = append(lines, visible...)
	return lines
}

// ---------------------------------------------------------------------------
// New-task modal (rendered in the main pane)

func (m model) renderInputModal(innerW, innerH int) []string {
	var lines []string
	header := "LAUNCH NEW TASK"
	if m.inputKind == "followup" {
		header = "FOLLOW-UP SESSION · " + shortRun(m.followupOf)
	}
	lines = append(lines, truncWidth(sectionStyle.Render(header), innerW), "")

	cursors := make([]string, 3)
	for i := range cursors {
		cursors[i] = "  "
	}
	if m.inputField >= 0 && m.inputField < 3 {
		cursors[m.inputField] = blueStyle.Render("> ")
	}
	issue := truncWidth(m.issueInput, max(1, innerW-16))
	repo := truncWidth(m.repoInput, max(1, innerW-16))
	profile := "(default)"
	hint := ""
	if m.modelCursor >= 0 && m.modelCursor < len(m.models) {
		mp := m.models[m.modelCursor]
		profile = fmt.Sprintf("%s · %s", mp.Profile, mp.Model)
		hint = fmt.Sprintf("  %d/%d profiles", m.modelCursor+1, len(m.models))
	}
	lines = append(lines, truncWidth(cursors[0]+midStyle.Render("issue: ")+textStyle.Render(issue), innerW))
	lines = append(lines, truncWidth(cursors[1]+midStyle.Render("repo:  ")+textStyle.Render(repo), innerW))
	if n := len(m.projects); n > 0 && m.projectCursor < n {
		lines = append(lines, truncWidth(
			"    "+dimStyle.Render(fmt.Sprintf("project %d/%d · [↑/↓] cycle", m.projectCursor+1, n)), innerW))
	}
	lines = append(lines, truncWidth(cursors[2]+midStyle.Render("model: ")+textStyle.Render(profile)+dimStyle.Render(hint), innerW))
	lines = append(lines, "")
	lines = append(lines, dimStyle.Render(" [tab] field · [↑/↓] pick · [enter] launch · [esc] cancel"))
	lines = append(lines, dimStyle.Render(" POST /api/tasks {issue, repo_root, model_profile, followup_of}"))
	return lines
}

// renderRunsTab is the chat session list: status glyph, title, id, outcome.
func (m model) renderRunsTab(innerW, innerH int) []string {
	var lines []string
	if m.replay != nil {
		lines = append(lines, dimStyle.Render(" sessions disabled in replay mode"))
		return lines
	}
	if m.runsStatus != "" {
		lines = append(lines, errStyle.Render(" runs: "+m.runsStatus))
		return lines
	}
	if len(m.runs) == 0 {
		lines = append(lines, dimStyle.Render(" no sessions yet — press [n] to start one"))
		return lines
	}
	lines = append(lines, truncWidth(sectionStyle.Render(fmt.Sprintf("SESSIONS (%d)", len(m.runs))), innerW), "")
	for i, run := range m.runs {
		glyph, style := "•", dimStyle
		switch run.Status {
		case "running":
			glyph, style = "▶", warnStyle
		case "verified":
			glyph, style = "✓", okStyle
		case "failed":
			glyph, style = "✗", errStyle
		}
		marker := " "
		if i == m.runsCursor {
			marker = blueStyle.Render("▸")
		}
		title := truncWidth(run.Title, max(10, innerW-46))
		right := fmt.Sprintf("%s %s", style.Render(glyph), dimStyle.Render(shortRun(run.RunID)))
		if run.ModelProfile != "" {
			right += dimStyle.Render(" · " + run.ModelProfile)
		}
		if run.FollowupOf != "" {
			right += dimStyle.Render(" ↩")
		}
		lines = append(lines, truncWidth(
			fmt.Sprintf("%s %s %s", marker, textStyle.Render(title), right), innerW))
		if i == m.runsCursor {
			detail := run.IssuePreview
			if detail == "" {
				detail = run.Outcome
			}
			if detail != "" {
				lines = append(lines, truncWidth("    "+dimStyle.Render(detail), innerW))
			}
			meta := run.RepoRoot
			if meta != "" {
				lines = append(lines, truncWidth("    "+dimStyle.Render(meta), innerW))
			}
		}
	}
	return windowLines(lines, max(1, innerH))
}

// ---------------------------------------------------------------------------
// Footer

func (m model) renderFooter() string {
	keys := "[1-6] tabs · [t] tree · [↑/↓/↵] agent · [n] new · [r] recon · [q] quit"
	if m.activeTab == tabRuns {
		keys = "[↑/↓] session · [↵] open · [f] follow-up · [x] stop · [d] delete · [n] new · [q] quit"
	}
	if m.replay != nil {
		keys = "[1-6] tabs · [t] tree · [↑/↓/↵] agent · [space] pause · [→] step · [q] quit"
	}
	if m.width < 100 {
		keys = "[1-6] tabs · [t] tree · [n] new · [r] recon · [q] quit"
		if m.replay != nil {
			keys = "[1-6] tabs · [t] tree · [space] pause · [→] step · [q] quit"
		}
	}
	focus := ""
	if m.focus == focusTree {
		focus = "  " + blueStyle.Render("◂ TREE")
	} else {
		focus = "  " + dimStyle.Render("▸ PANEL")
	}
	return truncWidth(footerStyle.Render(keys)+focus, m.width)
}

// ---------------------------------------------------------------------------
// Small render helpers

// tabBodyHeight estimates the scrollable height of the main tab body.
func tabBodyHeight(m model) int {
	bodyH := m.height - 2
	return bodyH - 2 - 2 // pane border + tab row + divider
}

// truncWidth hard-truncates a (possibly styled) line to w columns.
func truncWidth(s string, w int) string {
	if w <= 0 {
		return ""
	}
	if ansi.StringWidth(s) <= w {
		return s
	}
	return ansi.Truncate(s, w, "…")
}

// padRight pads a plain string to n columns (no truncation).
func padRight(s string, n int) string {
	if lipgloss.Width(s) >= n {
		return s
	}
	return s + strings.Repeat(" ", n-lipgloss.Width(s))
}

// padLines pads the block to exactly h lines.
func padLines(lines []string, h int) string {
	for len(lines) < h {
		lines = append(lines, "")
	}
	if len(lines) > h {
		lines = lines[:h]
	}
	return strings.Join(lines, "\n")
}

// windowLines keeps at most h lines (head-biased).
func windowLines(lines []string, h int) []string {
	if h <= 0 {
		return nil
	}
	if len(lines) <= h {
		return lines
	}
	return lines[:h]
}

// windowTail returns the last h lines adjusted by offset (0 = newest).
func windowTail(rows []string, offset, h int) []string {
	if h <= 0 {
		return nil
	}
	if len(rows) <= h {
		return rows
	}
	if offset > len(rows)-h {
		offset = len(rows) - h
	}
	return rows[len(rows)-h-offset : len(rows)-offset]
}

// windowHead returns h lines from offset (0 = first line).
func windowHead(rows []string, offset, h int) []string {
	if h <= 0 {
		return nil
	}
	if offset > len(rows) {
		offset = len(rows)
	}
	out := rows[offset:]
	if len(out) > h {
		out = out[:h]
	}
	return out
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
