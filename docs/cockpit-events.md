# Cockpit Event Contract (v2)

This is the contract between the engine (producer), the Go gateway (transport),
and the two cockpits (Web UI + Go TUI). Every event is one JSON object; live
transport is the gateway WebSocket (`/ws`), one JSON object per frame; offline
replay is one JSON object per line in a `.jsonl` file.

Common rules:

- Every event has `"event"` (kind) and `"run_id"` (the canonical run id).
- Digests are truncated (`<= 200` chars), never full file contents or outputs.
- Unknown kinds MUST be rendered generically (kind + run_id), never dropped.
- Consumers MUST tolerate missing optional fields.

## Agent hierarchy derivation

The roster comes from `GET /api/agents`:

```json
{"agents": [
  {"agent_id": "architect-1", "role": "architect", "model": "default",
   "specialties": ["architecture", "code-review"], "tool_tier": 3, "level": 1},
  {"agent_id": "mgr-1", "role": "manager", "model": "default",
   "specialties": ["coordination"], "tool_tier": 1, "level": 2},
  {"agent_id": "locator-1", "role": "locator", "model": "default",
   "specialties": ["localization", "code-navigation"], "tool_tier": 1, "level": 3},
  {"agent_id": "impl-1", "role": "implementer", "model": "default",
   "specialties": ["backend-api", "database", "frontend", "refactoring"], "tool_tier": 3, "level": 3},
  {"agent_id": "ver-1", "role": "verifier", "model": "default",
   "specialties": ["testing", "verification"], "tool_tier": 3, "level": 3}
]}
```

Levels: architect = L1 (top), manager = L2, configured specialists = L3.
Dynamic collaborators (`specialist.collaborator_added`) = L4 and attach to the
agent named in `"for"` (their id is `<parent>-collab-N`).

## 1. Run lifecycle

```json
{"event": "run.start", "run_id": "d1efdd2ac70f", "flags": [], "issue": "..."}
{"event": "run.end", "run_id": "d1efdd2ac70f", "success": true, "outcome": "VERIFIED: all tasks completed and gates passed"}
{"event": "run.failed", "run_id": "d1efdd2ac70f", "error": "...", "stage": "specialists"}
```

`run.start` carries the first 2000 chars of the issue. `run.failed` replaces
`run.end` on transport/configuration failure.

## 2. Architect phase

```json
{"event": "architect.profile", "run_id": "d1efdd2ac70f", "profile": {"languages": {...}, "test_framework": "pytest"}}
{"event": "architect.plan", "run_id": "d1efdd2ac70f", "reproduction_test": "tests/test_width_alignment.py",
 "subtasks": [
   {"id": "st-1", "title": "Implement fill/align/width semantics in parse()",
    "specialty": "bugfix", "complexity": 6, "files": ["parse/core.py"],
    "depends_on": [], "acceptance": "fill char, alignment and width honored...", "risk_notes": "..."},
   {"id": "st-2", "title": "Add tests for width edge cases", "specialty": "testing",
    "complexity": 3, "files": ["tests/test_width_alignment.py"], "depends_on": ["st-1"],
    "acceptance": "...", "risk_notes": ""}
 ]}
```

`subtasks` is a list of objects (v1 sent bare ids; v2 consumers MUST treat a
string entry as `{"id": <string>}` with everything else unknown).

## 3. Manager delegation

```json
{"event": "specialist.assigned", "run_id": "d1efdd2ac70f", "task": "st-1",
 "agent": "impl-1", "role": "implementer", "batch": 1,
 "routing": {"specialty": 0.4, "availability": 0.2, "load": 0.2, "capability": 0.2, "total": 0.8}}
{"event": "specialist.result", "run_id": "d1efdd2ac70f", "task": "st-1", "agent": "impl-1",
 "role": "implementer", "success": true, "summary": "...", "steps": 7, "tokens": 45210}
{"event": "specialist.collaborator_added", "run_id": "d1efdd2ac70f",
 "agent": "impl-1-collab-1", "for": "impl-1", "role": "implementer"}
```

`routing` is the 40/20/20/20 assignment score breakdown from DESIGN_SPEC §5.1
(each factor's contribution, `total` = sum). `batch` is the 1-based execution
batch number (batches run sequentially; members were file-disjoint).

## 4. Agent activity (new in v2)

```json
{"event": "agent.step", "run_id": "d1efdd2ac70f", "agent": "impl-1", "role": "implementer",
 "task": "st-1", "step": 3, "max_steps": 16, "phase": "thinking"}
{"event": "agent.tool", "run_id": "d1efdd2ac70f", "agent": "impl-1", "role": "implementer",
 "task": "st-1", "step": 3, "tool": "read_file", "args_digest": "parse/core.py",
 "ok": true, "duration_ms": 12, "result_digest": "842 lines"}
{"event": "agent.usage", "run_id": "d1efdd2ac70f", "agent": "impl-1", "task": "st-1",
 "prompt_tokens": 12345, "completion_tokens": 678, "total_tokens": 13023,
 "total_tokens_agent": 45210}
```

- `agent.step` fires at each model-call boundary (`phase`: "thinking" before
  the call, "responding" when the model answered). This is the "what is this
  agent doing right now" pulse.
- `agent.tool` fires after each tool execution. `args_digest` is the
  human-relevant argument (path / pattern / command / node id);
  `result_digest` is the truncated outcome.
- `agent.usage` fires after each model call; `total_tokens_agent` is that
  agent's cumulative total for the run (meters are set, not summed, from this).
- The architect's and manager's calls emit the same events with `task` =
  "planning" / "coordination" respectively.

## 5. Recovery ladder

```json
{"event": "recovery.l1_retry", "run_id": "d1efdd2ac70f", "task": "st-1", "agent": "impl-1",
 "attempt": 1, "error_type": "ValueError"}
{"event": "recovery.l2_guidance", "run_id": "d1efdd2ac70f", "task": "st-1", "agent": "impl-1",
 "guidance": "add collaborators"}
{"event": "recovery.l2_reroute", "run_id": "d1efdd2ac70f", "task": "st-1", "agent": "impl-1",
 "to": "impl-1-collab-1", "guidance": "add collaborators"}
{"event": "recovery.l3_replan", "run_id": "d1efdd2ac70f", "task": "st-1"}
{"event": "recovery.l3_skipped_budget", "run_id": "d1efdd2ac70f", "task": "st-1"}
```

`l2_guidance` = manager kept the executor; `l2_reroute` = executor swapped
(`to` names the replacement; with collaborators this is a spawn).

## 6. Verification & budget

```json
{"event": "baseline.captured", "run_id": "d1efdd2ac70f", "runnable": true,
 "pre_existing_failures": 0, "reproduction_test": "tests/test_width_alignment.py"}
{"event": "verification.stage", "run_id": "d1efdd2ac70f", "stage": "1-integrity",
 "passed": true, "blocking": true, "detail": "diff clean", "duration_seconds": 0.4}
{"event": "tokens.usage", "run_id": "d1efdd2ac70f", "phase": "specialists",
 "governor_mode": "NORMAL", "usage": {"prompt_tokens": 1900000, "completion_tokens": 210000,
 "total_tokens": 2110000}}
```

The six stages in order: `1-integrity, 2-self-check, 3-local-tests,
4-code-review, 5-security, 6-final-review`.

## Transport notes

- Gateway WS frames are the raw event objects (same shapes as above).
- Event history per run is capped (2000) and served by `GET /api/tasks/{run_id}`.
- The offline fixture `gateway/web/events.sample.jsonl` follows this contract
  exactly; both cockpits implement a replay mode against it.
