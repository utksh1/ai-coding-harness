// Foreman cockpit TUI — transport sources: the gateway WebSocket (live) and
// the JSONL replayer (offline verification mode).
package main

import (
	"bufio"
	"net/http"
	"os"
	"strings"
	"time"

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
