// Foreman cockpit TUI — the org-tree coding-agent cockpit.
//
// Left pane: the live agent hierarchy (L1 architect → L2 manager → L3
// specialists → L4 collaborators) with status lights, per-agent token
// meters, delegation scores and the current activity pulse. Right pane:
// Plan / Agent / Activity / Verify / Diff tabs. Live mode streams gateway
// WebSocket events; replay mode animates a cockpit-events JSONL fixture
// offline.
//
// Usage:
//
//	foreman-tui                                # live (GATEWAY_URL env, default ws://localhost:8080/ws)
//	foreman-tui --replay web/events.sample.jsonl [--replay-speed=150]
//
// Keys: [1-5] tabs · [t] tree focus · [↑/↓/↵] agent select · [n] new task
// (live) · [r] reconnect (live) · [space] pause · [→] step (replay) · [q] quit.
package main

import (
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/x/term"
)

func main() {
	replayPath := flag.String("replay", "", "replay a cockpit-events .jsonl fixture instead of connecting")
	replaySpeed := flag.Int("replay-speed", 150, "milliseconds per event in replay mode (0 = instant)")
	flag.Parse()

	gatewayURL := os.Getenv("GATEWAY_URL")
	if gatewayURL == "" {
		gatewayURL = "ws://localhost:8080/ws"
	}

	m := initialModel(gatewayURL)

	if *replayPath != "" {
		r, err := loadReplayer(*replayPath, time.Duration(*replaySpeed)*time.Millisecond)
		if err != nil {
			fmt.Fprintln(os.Stderr, "replay load failed:", err)
			os.Exit(1)
		}
		m.replay = r
		m.notice = fmt.Sprintf("replay: %s (%d events, %dms)", *replayPath, r.total(), *replaySpeed)
		if r.speed <= 0 {
			// Instant mode: fold the whole fixture in before the first frame.
			for {
				ev, ok := r.next()
				if !ok {
					break
				}
				m.state.Apply(ev)
			}
			m.notice = fmt.Sprintf("replay complete: %s (%d events)", *replayPath, r.total())
		}
	}

	opts := []tea.ProgramOption{}
	if !isTTY(os.Stdin) {
		m.headless = true
		opts = append(opts, tea.WithInput(nil))
	}
	if !isTTY(os.Stdout) {
		// Non-tty output: plain frames, no alt-screen takeover.
		opts = append(opts, tea.WithOutput(os.Stdout))
	} else {
		opts = append(opts, tea.WithAltScreen())
	}

	if _, err := tea.NewProgram(m, opts...).Run(); err != nil {
		fmt.Fprintln(os.Stderr, "tui error:", err)
		os.Exit(1)
	}
}

// initialModel builds the live-mode model (GATEWAY_URL → ws + http pair).
func initialModel(gatewayURL string) model {
	m := model{
		state:     NewState(),
		wsURL:     gatewayURL,
		apiURL:    apiURLFor(gatewayURL),
		msgs:      make(chan wsFrame, 256),
		width:     120,
		height:    40,
		repoInput: "fixtures/mini-repo",
	}
	return m
}

// apiURLFor derives the HTTP base from the WebSocket URL.
func apiURLFor(gatewayURL string) string {
	api := gatewayURL
	switch {
	case strings.HasPrefix(api, "ws://"):
		api = "http://" + strings.TrimPrefix(api, "ws://")
	case strings.HasPrefix(api, "wss://"):
		api = "https://" + strings.TrimPrefix(api, "wss://")
	case strings.HasPrefix(api, "http://"):
	case strings.HasPrefix(api, "https://"):
	default:
		api = "http://" + api
	}
	// Strip the path component ONLY after the scheme's authority: a naive
	// Index("/") hits the slash inside "http://" and yields "http:" (live
	// finding: the roster fetch silently failed, tree stayed "derived").
	if i := strings.Index(api, "://"); i != -1 {
		if j := strings.Index(api[i+3:], "/"); j != -1 {
			api = api[:i+3+j]
		}
	}
	return api
}

// isTTY reports whether f is an interactive terminal (NOT merely a char
// device: /dev/null is one too and must not count — bubbletea would then try
// to open /dev/tty, which headless environments lack).
func isTTY(f *os.File) bool {
	return term.IsTerminal(f.Fd())
}
