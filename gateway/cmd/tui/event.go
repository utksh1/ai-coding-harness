// Foreman cockpit TUI — event parsing (pure, no UI deps).
//
// One tolerant struct decodes every kind from docs/cockpit-events.md v2.
// Unknown kinds and missing fields never error: consumers render what is
// there and default the rest.
package main

import "encoding/json"

// Event is the union of every known cockpit event kind. Fields absent from
// the wire JSON stay zero-valued; booleans that matter use pointers so a
// missing field is distinguishable from an explicit false.
type Event struct {
	Kind  string `json:"event"`
	RunID string `json:"run_id"`

	// Run lifecycle.
	Success *bool    `json:"success"`
	Outcome string   `json:"outcome"`
	Error   string   `json:"error"`
	Stage   string   `json:"stage"` // run.failed stage
	Flags   []string `json:"flags"`
	Issue   string   `json:"issue"`

	// Architect phase.
	Profile   *ProfileEvent `json:"profile"`
	ReproTest string        `json:"reproduction_test"`
	Subtasks  SubtaskList   `json:"subtasks"`

	// Manager delegation.
	Task    string   `json:"task"`
	Agent   string   `json:"agent"`
	Role    string   `json:"role"`
	Batch   int      `json:"batch"`
	Routing *Routing `json:"routing"`
	Summary string   `json:"summary"`
	Steps   int      `json:"steps"`
	Tokens  int      `json:"tokens"` // specialist.result task total
	For     string   `json:"for"`    // collaborator parent

	// Agent activity (v2).
	Step         int    `json:"step"`
	MaxSteps     int    `json:"max_steps"`
	Phase        string `json:"phase"`
	Tool         string `json:"tool"`
	ArgsDigest   string `json:"args_digest"`
	OK           *bool  `json:"ok"`
	DurationMS   int    `json:"duration_ms"`
	ResultDigest string `json:"result_digest"`

	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
	// total_tokens_agent is the agent's cumulative meter (SET semantics).
	TotalTokensAgent int `json:"total_tokens_agent"`

	// Recovery ladder.
	Attempt   int    `json:"attempt"`
	ErrorType string `json:"error_type"`
	Guidance  string `json:"guidance"`
	To        string `json:"to"` // l2_reroute target

	// Verification & budget.
	Passed          *bool   `json:"passed"`
	Blocking        *bool   `json:"blocking"`
	Skipped         *bool   `json:"skipped"`
	Detail          string  `json:"detail"`
	DurationSeconds float64 `json:"duration_seconds"`

	// baseline.captured.
	Runnable            *bool `json:"runnable"`
	PreExistingFailures int   `json:"pre_existing_failures"`

	// tokens.usage.
	GovernorMode string     `json:"governor_mode"`
	Usage        *UsageBody `json:"usage"`
}

// ProfileEvent is architect.profile's repo fingerprint.
type ProfileEvent struct {
	Languages     map[string]int `json:"languages"`
	TestFramework string         `json:"test_framework"`
	Files         int            `json:"files"`
	LOC           int            `json:"loc"`
}

// Routing is the 40/20/20/20 assignment score breakdown.
type Routing struct {
	Specialty    float64 `json:"specialty"`
	Availability float64 `json:"availability"`
	Load         float64 `json:"load"`
	Capability   float64 `json:"capability"`
	Total        float64 `json:"total"`
}

// UsageBody is the tokens.usage meter (cumulative, SET semantics).
type UsageBody struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// SubtaskSpec is one architect.plan subtask. The v1 wire format sent bare
// ids; a string decodes as {"id": s} with everything else unknown.
type SubtaskSpec struct {
	ID         string
	Title      string
	Specialty  string
	Complexity int
	Files      []string
	DependsOn  []string
	Acceptance string
	RiskNotes  string
}

// SubtaskList tolerates a JSON array whose entries are objects or strings.
type SubtaskList []SubtaskSpec

// UnmarshalJSON accepts both v2 objects and v1 bare-id strings.
func (l *SubtaskList) UnmarshalJSON(data []byte) error {
	var raw []json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		// Tolerate a non-array (or missing) value: treat as empty.
		*l = nil
		return nil
	}
	out := make(SubtaskList, 0, len(raw))
	for _, item := range raw {
		var s string
		if err := json.Unmarshal(item, &s); err == nil {
			out = append(out, SubtaskSpec{ID: s})
			continue
		}
		var spec struct {
			ID         string   `json:"id"`
			Title      string   `json:"title"`
			Specialty  string   `json:"specialty"`
			Complexity int      `json:"complexity"`
			Files      []string `json:"files"`
			DependsOn  []string `json:"depends_on"`
			Acceptance string   `json:"acceptance"`
			RiskNotes  string   `json:"risk_notes"`
		}
		if err := json.Unmarshal(item, &spec); err != nil {
			continue // one malformed entry never poisons the plan
		}
		out = append(out, SubtaskSpec(spec))
	}
	*l = out
	return nil
}

// RosterAgent is one entry of GET /api/agents (all fields optional).
type RosterAgent struct {
	AgentID     string   `json:"agent_id"`
	ID          string   `json:"id"` // tolerated alias
	Role        string   `json:"role"`
	Model       string   `json:"model"`
	Specialties []string `json:"specialties"`
	ToolTier    int      `json:"tool_tier"`
	Level       int      `json:"level"`
}

// RosterResponse is the /api/agents envelope.
type RosterResponse struct {
	Agents []RosterAgent `json:"agents"`
}

// EvidenceResponse is GET /api/evidence/{run}/file/patch.diff (content is the
// file body; found=false means the run has no patch yet).
type EvidenceResponse struct {
	Found   bool   `json:"found"`
	RunID   string `json:"run_id"`
	Name    string `json:"name"`
	Content string `json:"content"`
}

// parseEvent decodes one cockpit event line. A JSON-decode failure is an
// error the caller may log and skip — never a crash.
func parseEvent(line []byte) (Event, error) {
	var ev Event
	err := json.Unmarshal(line, &ev)
	return ev, err
}
