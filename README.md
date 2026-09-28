# LCC x DevClub AI Coding Harness - Architecture Documentation

> **Autonomous Coding Agent Harness for Software Engineering Tasks**  
> A hierarchical multi-agent system with intelligent orchestration, error recovery, and verification pipelines.

---

## Source of truth

1. **`docs/specs/foreman-eval-mode-design.md`** — the approved runtime design (eval contract, architecture, build order).
2. **The code in `src/harness/`** — when an older document disagrees with the code, the code wins.
3. `DESIGN_SPEC.md` / `TECHNICAL_IMPLEMENTATION.md` are **legacy vision documents**: read for surviving concepts, never as build instructions (see `CONTEXT_FOR_AI.md`).

---

## Quickstart (standard evaluation interface)

```bash
export AI_API_KEY="<PROVIDED_API_KEY>"   # credential is supplied at runtime, never committed
make setup    # create .venv, install pinned dependencies, verify environment
make run      # health summary + latest-evidence pointer (headless by design)
make test     # offline test suite with coverage
make lint     # ruff (black/flake8-compatible) + mypy type check
```

Enable lint-on-commit hooks once: `pip install pre-commit && pre-commit install`.

## The `foreman` launcher (the product surface)

One command drives the whole platform — services, cockpits, projects, and
chat-style runs. After `make setup && make install`, `foreman` is on PATH:

```bash
make install          # builds the Go cockpit binaries + links ~/.local/bin/foreman

foreman doctor        # environment + service diagnostics (honest, exit-coded)
foreman start --demo  # orchestrator :8000 + gateway :8080, daemonized & supervised
                      #   --demo runs scripted model responses — no API key needed
foreman web           # the web cockpit (chat sessions, org tree, gates, diff)
foreman tui           # the Go TUI cockpit (same contract, in the terminal)

foreman projects add ~/code/my-app        # register a repo (git state enriched)
foreman run "fix the off-by-one in parser" --repo ~/code/my-app --model luna
                      # submits a run through the gateway; prints the run id
                      #   --wait    block until the verdict event
                      #   --demo    scripted responses (offline demonstration)
foreman followup <RUN_ID> "now add tests" # continue THAT session: prior summary
                      #   + patch footprint prepended, same repo, chat-style
foreman cancel <RUN_ID>                   # honest stop: run.end(stop_reason=cancelled)
foreman runs           # session list (✔ verified · ✘ failed · ■ cancelled · ▶ running)
foreman status         # health, versions, journal size, live session summary
foreman logs gateway   # tail service logs (orchestrator | gateway | journal)
foreman stop           # graceful stop (SIGTERM → escalate → orphan sweep)
```

Operational guarantees baked into the launcher (each from a real incident):

- **Services survive the launching shell** — daemons are `setsid -f` with
  pid files written by the daemon itself, so a closed terminal (or a
  crashed launcher) never orphans a run mid-flight.
- **The session list survives restarts** — the gateway journals every event
  (`logs/gateway-events.jsonl`, 50 MiB rotation) and replays on boot; after
  `foreman restart`, `foreman runs` shows every session with its verdict.
- **One run per repo at a time** — the admission guard rejects concurrent
  pipelines on one working tree (interleaved edits corrupted a live run once).
- **Stops are honest** — cancel emits `run.failed("CANCELLED by user")` +
  `run.end(success=false, stop_reason="cancelled")` + a `cancelled` session
  state, never a silent drop or a fake failure.

### Model providers

`harness.yaml` holds named model profiles; agents reference them by key, so a
provider switch is a one-line change. Supported: `openai-compatible` (any
OpenAI-style endpoint — proxies, vLLM, OpenRouter), `google` (Gemini via
generateContent, native function calling), `openai`, `anthropic`, `fake`
(offline tests). Two operational notes from live runs:

- **Gemini is region-restricted**: `generativelanguage.googleapis.com`
  refuses unsupported egress regions (`FAILED_PRECONDITION "User location is
  not supported"`). Express-mode keys (`AQ.Ab…`) additionally need the Agent
  Platform API enabled in the owning GCP project.
- **No external quota?** A loopback OpenAI-compatible bridge over
  `z-ai-web-dev-sdk` ships in `tools/dev/` — `bun tools/dev/zai_openai_shim.mjs`
  listens on `127.0.0.1:8788` with native function calling, upstream
  throttling and 429 backoff built in. Point the default profile at
  `http://127.0.0.1:8788/v1` (see `tools/dev/README.md`).

Budget rails: every run is bounded by BOTH `budget.total_tokens` (token
governor with NORMAL→SURGICAL→FINALIZE modes) and `run.wall_clock_seconds`
(a monotonic deadline checked on every model call and stage boundary — a
throttled provider can starve a run without spending tokens, so the clock
trips first and finalizes honestly with a `budget.exhausted` event).

## Platform layer (beyond the hackathon — epic #69)

The engine is a clean importable library, so the multi-service platform from
`TECHNICAL_IMPLEMENTATION.md` wraps it without touching the graded core.
**`foreman` (above) is the intended entry point** — the targets below are the
unpackaged equivalents, useful in CI and development:

```bash
make build          # build both Go binaries without starting anything
make gateway        # Go API gateway (:8080) — proxies to the orchestrator, WS broadcast
make tui-go         # Go TUI cockpit (Bubble Tea) consuming the gateway
make go-test        # Go unit tests
docker compose -f platform/docker-compose.yml up --build   # redis + orchestrator + gateway
```

- **Orchestrator** (`harness[platform]` extra, port 8000): FastAPI service exposing
  `/agent/architect/{analyze,decompose}`, `/agent/manager/assign`,
  `/agent/specialist/execute`, `/agent/run`, `/agent/status/:id`
- **Gateway** (Go, port 8080): async task creation (202 + run_id), `/api/tasks/:id`
  with full stored event history, `/api/agents`, `/api/evidence/{run}/file/{name}`,
  `/ws` WebSocket with backlog replay (a mid-run refresh re-renders the whole run)

**The cockpits** (the operator's window into a live run — both consume the
[Cockpit Event Contract v2](docs/cockpit-events.md)):

- **Go TUI** (`make tui-go`): the org-tree cockpit — L1 architect → L2 manager →
  L3 specialists → L4 collaborators with status lights, per-agent token meters,
  step counters and live activity; tabs for the plan board, per-agent tool-call
  log, activity feed, six verification gates and the final diff. Offline replay:
  `go run ./cmd/tui --replay web/events.sample-collab.jsonl`
- **Web cockpit** (`http://localhost:8080/`): the same hierarchy in the browser —
  kanban plan board, delegation routing breakdown (40/20/20/20), tool-call log,
  timeline with auto-scroll, gate stepper, diff viewer, launcher.
  Offline replay: `http://localhost:8080/?replay=events.luna-run.jsonl`

**Engine quality loops** (what keeps a flailing model on course — live-run
findings, each backed by tests):

- **Loop coaching**: repeated identical tool-call failures inject the correct
  usage contract; step-budget warnings at 70%/90%; edits not followed by a test
  run get a verify nudge; identical failing calls are deduped.
- **Self-recovering `apply_edit`**: a failed search/replace returns the closest
  actual file region (line-numbered) so the model corrects itself in one
  round-trip instead of read → guess → fail loops.
- **Architect-stage resilience**: transient provider outages at t=0 are retried
  with backoff (`architect.retry` events); auth errors fail fast.
- **Recovery ladder**: L1 self-repair (3×) → L2 manager guidance / re-route /
  collaborator spawn → L3 architect re-plan → honest `NOT VERIFIED` with evidence.

Never imported by the graded core: `make setup/run/test` stay exactly as evaluated.

**Supplying the issue to `make run`** (all protocols supported — see
[docs/eval-runbook.md](docs/eval-runbook.md)):

```bash
echo "the issue text" | make run          # piped stdin
HARNESS_ISSUE_FILE=issue.md make run      # file
harness solve --issue "..." --repo /path/to/target   # direct
harness replay                            # inspect a finished run offline
```

---

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [Agent Workflow Pipeline](#2-agent-workflow-pipeline)
3. [Error Recovery & Escalation](#3-error-recovery--escalation)
4. [Task Assignment Algorithm](#4-task-assignment-algorithm)
5. [Context Management](#5-context-management)
6. [Git Workflow](#6-git-workflow)
7. [Tool Ecosystem](#7-tool-ecosystem)
8. [Resource Management](#8-resource-management)
9. [Security Architecture](#9-security-architecture)
10. [Verification Pipeline](#10-verification-pipeline)

---

## 1. System Architecture Overview

### 1.1 Five-Layer Architecture

The system is organized into five distinct layers, each with specific responsibilities:

```mermaid
graph TB
    subgraph Layer1[" "]
        direction LR
        UI1["Web Dashboard"]
        UI2["Configuration Interface"]
        UI3["Testing Console"]
        UI4["Monitoring System"]
        UI5["REST API"]
    end
    
    subgraph Layer2[" "]
        Architect["ARCHITECT AGENT<br/>────────────────<br/>• Repository Analysis<br/>• Task Decomposition<br/>• Global Rule Extraction<br/>• GitHub Issue Creation<br/>• Final PR Review<br/>────────────────<br/>Model: GPT-4 / Claude Opus<br/>Temperature: 0.2"]
    end
    
    subgraph Layer3[" "]
        M1["MANAGER: Backend<br/>──────────────<br/>Coordinates 8 specialists<br/>Backend API, Database,<br/>Authentication, Caching"]
        M2["MANAGER: Frontend<br/>──────────────<br/>Coordinates 6 specialists<br/>React, Vue, Styling,<br/>Testing"]
        M3["MANAGER: DevOps<br/>──────────────<br/>Coordinates 4 specialists<br/>CI/CD, Infrastructure,<br/>Security"]
    end
    
    subgraph Layer4[" "]
        S1["Backend-API<br/>Specialist"]
        S2["Database<br/>Specialist"]
        S3["React<br/>Specialist"]
        S4["DevOps<br/>Specialist"]
        S5["Security<br/>Specialist"]
        S6["Testing<br/>Specialist"]
    end
    
    subgraph Layer5[" "]
        I1["Context Store<br/>PostgreSQL"]
        I2["Tool Runtime<br/>23+ Tools"]
        I3["Model APIs<br/>Multi-Provider"]
        I4["GitHub<br/>Integration"]
    end
    
    Layer1 --> Architect
    Architect --> M1
    Architect --> M2
    Architect --> M3
    M1 --> S1
    M1 --> S2
    M1 --> S6
    M2 --> S3
    M3 --> S4
    M3 --> S5
    Layer4 -.-> Layer5
    
    classDef default stroke:#333,stroke-width:2px
```

**Layer Descriptions:**

- **Layer 1 - User Interface**: All user-facing components for interaction, monitoring, and configuration
- **Layer 2 - Orchestration**: Single architect agent responsible for high-level coordination
- **Layer 3 - Coordination**: Manager agents that route tasks and monitor specialist progress
- **Layer 4 - Execution**: Specialist agents that implement features, write tests, and create pull requests
- **Layer 5 - Infrastructure**: Supporting services for persistence, tools, models, and version control

---

## 2. Agent Workflow Pipeline

### 2.1 Eleven-Step Specialist Execution Flow

Each specialist follows a standardized workflow from task receipt to integration:

```mermaid
flowchart LR
    Start([Task Assigned]) --> S1[1. Receive Task<br/>& Context]
    S1 --> S2[2. Create Branch<br/>agent/id/issue-desc]
    S2 --> S3[3. Implement<br/>Code + Tests]
    S3 --> S4[4. Self-Verify<br/>Run Tests Locally]
    
    S4 --> Decision{Tests<br/>Pass?}
    Decision -->|No| S5[5. Error Recovery<br/>Max 3 Attempts]
    S5 --> Retry{Fixed?}
    Retry -->|Yes| S6
    Retry -->|No| Escalate[Escalate to<br/>Manager]
    Decision -->|Yes| S6[6. Commit & Push]
    
    S6 --> S7[7. Create Pull Request]
    S7 --> S8[8. CI/CD Checks<br/>Automated]
    S8 --> S9[9. Code Review<br/>Optional Agent]
    S9 --> S10[10. Manager Review<br/>Conflict Check]
    S10 --> S11[11. Architect Review<br/>& Merge]
    S11 --> End([Integration<br/>Complete])
    
    classDef default stroke:#333,stroke-width:2px
```

**Key Characteristics:**

1. **Isolation**: Each specialist works in a dedicated branch
2. **Self-Verification**: Agents validate their work before submission
3. **Progressive Review**: Multiple validation stages ensure quality
4. **Error Recovery**: Built-in retry mechanism with escalation path

---

## 3. Error Recovery & Escalation

### 3.1 Four-Level Recovery Hierarchy

The system provides graduated error handling with increasing human involvement:

```mermaid
graph TD
    Error["ERROR DETECTED<br/>──────────────<br/>Test Failure | API Timeout<br/>Invalid Output | Tool Error"]
    
    L1["LEVEL 1: Specialist Self-Recovery<br/>────────────────────────────<br/>Attempt 1: Parse error message, fix syntax<br/>Attempt 2: Debug logic, alternative fix<br/>Attempt 3: Different approach entirely<br/>────────────────────────────<br/>Constraint: Maximum 3 attempts or 30 minutes"]
    
    L2["LEVEL 2: Manager Intervention<br/>──────────────────────────<br/>Analysis:<br/>• Skill Gap → Reassign to different specialist<br/>• Tool Limitation → Grant access or upgrade model<br/>• Complex Task → Assign 2-3 specialists to collaborate<br/>• Unclear Requirements → Reframe task with clarity<br/>──────────────────────────<br/>Monitoring: Token consumption vs. progress ratio"]
    
    L3["LEVEL 3: Architect Escalation<br/>────────────────────────<br/>Strategic Review:<br/>• Task specification unclear → Reformulate completely<br/>• Task too complex → Decompose into subtasks<br/>• Missing repository context → Perform deep analysis<br/>• Conflicting requirements → Request user decision<br/>────────────────────────<br/>Authority: Can reassign to different manager team"]
    
    L4["LEVEL 4: Human Intervention<br/>───────────────────────<br/>Required When:<br/>• Architect cannot resolve after analysis<br/>• External system or infrastructure failure<br/>• User decision needed on trade-offs<br/>• Budget constraints exceeded<br/>• Security judgment required<br/>───────────────────────<br/>Notification: Dashboard alert + optional email/Slack"]
    
    Success1["✓ Resolved<br/>Continue"]
    Success2["✓ Resolved<br/>Continue"]
    Success3["✓ Resolved<br/>Continue"]
    
    Error --> L1
    L1 -->|Success| Success1
    L1 -->|3 Failures| L2
    L2 -->|Success| Success2
    L2 -->|2 Failures| L3
    L3 -->|Success| Success3
    L3 -->|Cannot Resolve| L4
    
    classDef default stroke:#333,stroke-width:2px
```

**Escalation Criteria:**

- **Level 1 → 2**: After 3 failed attempts or 30 minutes elapsed
- **Level 2 → 3**: After 2 manager intervention attempts fail
- **Level 3 → 4**: When architect determines human decision required

---

## 4. Task Assignment Algorithm

### 4.1 Multi-Factor Scoring System

Managers use a weighted scoring algorithm to assign tasks optimally:

```mermaid
graph LR
    Task["INCOMING TASK<br/>─────────────<br/>Complexity: 7/10<br/>Specialty: Backend API<br/>Priority: High<br/>Required Tools:<br/>filesystem, git, database"]
    
    F1["Factor 1<br/>SPECIALTY MATCH<br/>─────────────<br/>Weight: 40%<br/>─────────────<br/>Measures domain expertise<br/>alignment with task<br/>requirements"]
    
    F2["Factor 2<br/>AVAILABILITY<br/>─────────────<br/>Weight: 20%<br/>─────────────<br/>Current task load<br/>and queue length"]
    
    F3["Factor 3<br/>LOAD BALANCE<br/>─────────────<br/>Weight: 20%<br/>─────────────<br/>Token consumption<br/>vs team average"]
    
    F4["Factor 4<br/>CAPABILITY<br/>─────────────<br/>Weight: 20%<br/>─────────────<br/>Model tier and<br/>tool permissions"]
    
    Calc["CALCULATION<br/>──────────────<br/>For each specialist:<br/>──────────────<br/>Score = <br/>specialty × 0.4<br/>+ availability × 0.2<br/>+ load_balance × 0.2<br/>+ capability × 0.2"]
    
    Decision["ASSIGNMENT<br/>────────────<br/>If complexity ≤ 7:<br/>Assign highest scorer<br/>────────────<br/>If complexity > 7:<br/>Assign top 2-3<br/>for collaboration"]
    
    Task --> F1
    Task --> F2
    Task --> F3
    Task --> F4
    F1 --> Calc
    F2 --> Calc
    F3 --> Calc
    F4 --> Calc
    Calc --> Decision
    
    classDef default stroke:#333,stroke-width:2px
```

**Scoring Details:**

- **Specialty Match (40%)**: Historical success rate on similar tasks
- **Availability (20%)**: 100 if idle, decreases with active tasks
- **Load Balance (20%)**: Relative to team average token consumption
- **Capability (20%)**: Model tier and required tool access

---

## 5. Context Management

### 5.1 Hierarchical Context Architecture

Context is organized globally and per-agent with time-based compression:

```mermaid
graph TB
    subgraph Global["GLOBAL CONTEXT STORE"]
        Repo["Repository Metadata<br/>───────────────<br/>• Structure & organization<br/>• Dependencies<br/>• Build commands<br/>• Test framework"]
        Tasks["Task Registry<br/>───────────────<br/>• All active tasks<br/>• Status tracking<br/>• Dependency graph<br/>• Priority levels"]
        Agents["Agent Registry<br/>───────────────<br/>• Agent availability<br/>• Current assignments<br/>• Token consumption<br/>• Performance metrics"]
        Rules["Global Rules<br/>───────────────<br/>• Coding standards<br/>• Architecture decisions<br/>• User preferences<br/>• Constraints"]
    end
    
    subgraph Agent1["AGENT CONTEXT: Backend-API-1"]
        W1["Window 1: RECENT<br/>──────────────<br/>Last 20 messages<br/>Full detail preserved<br/>Direct access"]
        W2["Window 2: MID-TERM<br/>──────────────<br/>Compressed summaries<br/>Key decisions retained<br/>Retrieved on demand"]
        W3["Window 3: HISTORICAL<br/>──────────────<br/>High-level outcomes<br/>Major milestones only<br/>Archived"]
    end
    
    subgraph Agent2["AGENT CONTEXT: React-1"]
        W1_2["Window 1: RECENT<br/>──────────────<br/>Last 20 messages<br/>Full detail preserved<br/>Direct access"]
        W2_2["Window 2: MID-TERM<br/>──────────────<br/>Compressed summaries<br/>Key decisions retained<br/>Retrieved on demand"]
        W3_2["Window 3: HISTORICAL<br/>──────────────<br/>High-level outcomes<br/>Major milestones only<br/>Archived"]
    end
    
    Retrieval["RETRIEVAL STRATEGY<br/>──────────────────<br/>1. Check Window 1 (recent)<br/>2. Expand to Window 2 if needed<br/>3. Check neighbor agent contexts<br/>4. Retrieve historical if critical<br/>──────────────────<br/>Neighbor = related tasks or files"]
    
    Global -.-> Agent1
    Global -.-> Agent2
    Agent1 -.-> Agent2
    Agent1 --> Retrieval
    Agent2 --> Retrieval
    
    classDef default stroke:#333,stroke-width:2px
```

**Compression Strategy:**

- **Window 1**: No compression, immediate access
- **Window 2**: Compress every 10 messages into summary
- **Window 3**: Compress every 20 summaries into milestone list
- **Neighbor Access**: Related tasks share compressed contexts

---

## 6. Git Workflow

### 6.1 Branch-Per-Agent Strategy

Parallel development with conflict prevention through branch isolation:

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'git0': '#333', 'git1': '#666', 'git2': '#999', 'gitBranchLabel0': '#fff', 'gitBranchLabel1': '#fff', 'gitBranchLabel2': '#fff', 'commitLabelColor': '#000', 'commitLabelBackground': '#fff'}}}%%
gitGraph
    commit id: "Initial: Repository setup"
    commit id: "feat: Core architecture"
    
    branch agent/sd1/42-add-auth
    checkout agent/sd1/42-add-auth
    commit id: "feat: User model"
    commit id: "feat: JWT middleware"
    commit id: "test: Auth tests"
    
    checkout main
    branch agent/sd2/43-login-ui
    checkout agent/sd2/43-login-ui
    commit id: "feat: Login component"
    commit id: "style: Login UI"
    
    checkout main
    branch agent/sd3/44-e2e-tests
    checkout agent/sd3/44-e2e-tests
    commit id: "test: E2E auth flow"
    
    checkout main
    merge agent/sd1/42-add-auth tag: "Merge: Architect approved"
    
    checkout main
    merge agent/sd2/43-login-ui tag: "Merge: Architect approved"
    
    checkout main
    merge agent/sd3/44-e2e-tests tag: "Merge: Architect approved"
    
    commit id: "Release: v1.0.0"
```

**Branch Naming Convention:**
```
agent/<agent-id>/<issue-number>-<short-description>

Examples:
- agent/sd-backend-1/42-add-authentication
- agent/sd-react-1/43-create-login-ui
- agent/sd-testing-1/44-e2e-auth-tests
```

**Protection Rules:**
- `main`: Architect-only merge access, no force-push
- `agent/*`: Specialist can push, manager can coordinate
- All merges require passing CI/CD checks
- Branches auto-delete after successful merge

---

## 7. Tool Ecosystem

### 7.1 Three-Tier Access Control

Tools are organized by capability requirements and model sophistication:

```mermaid
graph TB
    subgraph Tier1["TIER 1: Basic Operations"]
        T1_1["filesystem_read<br/>─────────────<br/>Read file contents<br/>with line ranges"]
        T1_2["filesystem_list<br/>─────────────<br/>List directory<br/>contents"]
        T1_3["git_status<br/>─────────────<br/>Check working<br/>directory status"]
        T1_4["git_log<br/>─────────────<br/>View commit<br/>history"]
        T1_5["logging<br/>─────────────<br/>Write structured<br/>logs"]
    end
    
    subgraph Tier2["TIER 2: Development Operations"]
        T2_1["filesystem_write<br/>─────────────<br/>Modify files with<br/>backup creation"]
        T2_2["git_operations<br/>─────────────<br/>Branch, commit,<br/>push operations"]
        T2_3["grep_search<br/>─────────────<br/>Pattern search<br/>in codebase"]
        T2_4["npm_commands<br/>─────────────<br/>Package manager<br/>operations"]
        T2_5["test_runner<br/>─────────────<br/>Execute test<br/>suites"]
        T2_6["linting<br/>─────────────<br/>Code quality<br/>analysis"]
        T2_7["formatting<br/>─────────────<br/>Code formatting<br/>enforcement"]
    end
    
    subgraph Tier3["TIER 3: Advanced Operations"]
        T3_1["database_client<br/>─────────────<br/>Database queries<br/>with safety checks"]
        T3_2["api_testing<br/>─────────────<br/>HTTP request<br/>testing"]
        T3_3["code_execution<br/>─────────────<br/>Sandboxed code<br/>execution"]
        T3_4["migration_tools<br/>─────────────<br/>Database schema<br/>migrations"]
        T3_5["docker_commands<br/>─────────────<br/>Container<br/>operations"]
        T3_6["security_scanners<br/>─────────────<br/>Vulnerability<br/>detection"]
        T3_7["performance_profiling<br/>─────────────<br/>Performance<br/>analysis"]
        T3_8["github_api<br/>─────────────<br/>Issues, PRs,<br/>merge operations"]
    end
    
    Models["MODEL TIER MAPPING<br/>────────────────────<br/>Tier 1 Models → Tier 1 Tools<br/>GPT-3.5, Gemini Flash, Llama 3 8B<br/>────────────────────<br/>Tier 2 Models → Tier 1 + 2 Tools<br/>GPT-4-mini, Gemini Flash, Mistral<br/>────────────────────<br/>Tier 3 Models → All Tools<br/>GPT-4, Claude Sonnet, Gemini Pro<br/>────────────────────<br/>Tier 4 Models → All Tools + Architecture<br/>GPT-4, Claude Opus (Architect only)"]
    
    Sandbox["SANDBOX ENVIRONMENT<br/>───────────────────<br/>Isolation: Containerized execution<br/>Network: No external access<br/>CPU: 30 second limit<br/>Memory: 512 MB limit<br/>Filesystem: Temporary only<br/>───────────────────<br/>Applies to: code_execution tool"]
    
    Models -.-> Tier1
    Models -.-> Tier2
    Models -.-> Tier3
    Tier3 -.-> Sandbox
    
    classDef default stroke:#333,stroke-width:2px
```

**Access Control Logic:**
```
if agent.model_tier >= tool.required_tier:
    grant_access()
else:
    deny_with_suggestion_to_upgrade()
```

---

## 8. Resource Management

### 8.1 Budget Allocation Strategies

Four approaches for token budget management, selectable by user:

```mermaid
graph TB
    Budget["GLOBAL TOKEN BUDGET<br/>──────────────────<br/>Example: 1,000,000 tokens/day<br/>──────────────────<br/>User configures allocation strategy"]
    
    S1["Strategy 1: FIXED PER-AGENT<br/>──────────────────────────<br/>Architect: Unlimited (critical path)<br/>Each Manager: 100,000 tokens<br/>Each Specialist: 50,000 tokens<br/>──────────────────────────<br/>Advantages:<br/>• Predictable costs<br/>• Simple to understand<br/>• Easy budget planning<br/>──────────────────────────<br/>Disadvantages:<br/>• Inflexible allocation<br/>• Idle agents waste budget<br/>• Cannot adapt to workload"]
    
    S2["Strategy 2: DYNAMIC REALLOCATION<br/>─────────────────────────────<br/>Base allocation: 50,000 per agent<br/>Track efficiency: tasks / tokens<br/>──────────────────────────<br/>Every 6 hours:<br/>High performers (+20% budget)<br/>Average performers (no change)<br/>Low performers (-10% budget)<br/>─────────────────────────────<br/>Advantages:<br/>• Rewards productivity<br/>• Self-optimizing<br/>• Maximizes output per token<br/>─────────────────────────────<br/>Disadvantages:<br/>• More complex logic<br/>• Can penalize struggling agents"]
    
    S3["Strategy 3: PRIORITY-BASED<br/>──────────────────────────<br/>Budget allocated per task:<br/>Critical priority: 100,000 tokens<br/>High priority: 50,000 tokens<br/>Normal priority: 30,000 tokens<br/>Low priority: 15,000 tokens<br/>──────────────────────────<br/>Advantages:<br/>• Aligned with business value<br/>• Critical work never blocked<br/>• Clear prioritization<br/>──────────────────────────<br/>Disadvantages:<br/>• Hard to predict total cost<br/>• Requires careful priority setting"]
    
    S4["Strategy 4: ADAPTIVE POOL<br/>─────────────────────────<br/>Shared pool, agents draw as needed<br/>No per-agent limits<br/>─────────────────────────<br/>Pool level responses:<br/>80% consumed: Prioritize high-value<br/>90% consumed: Critical tasks only<br/>100% consumed: Pause all work<br/>─────────────────────────<br/>Advantages:<br/>• Maximum flexibility<br/>• No artificial constraints<br/>• Natural prioritization<br/>─────────────────────────<br/>Disadvantages:<br/>• Unpredictable consumption<br/>• Risk of early exhaustion"]
    
    Monitor["MONITORING & OPTIMIZATION<br/>───────────────────────────<br/>Tracked Metrics:<br/>• Tokens consumed per agent<br/>• Tasks completed per agent<br/>• Efficiency ratio (tasks/tokens)<br/>• Cost per task ($)<br/>• Time per task<br/>───────────────────────────<br/>Automated Recommendations:<br/>• Downgrade models for simple tasks<br/>• Upgrade models for struggling agents<br/>• Reallocate budget to high performers<br/>• Identify inefficient task types"]
    
    Budget --> S1
    Budget --> S2
    Budget --> S3
    Budget --> S4
    S1 --> Monitor
    S2 --> Monitor
    S3 --> Monitor
    S4 --> Monitor
    
    classDef default stroke:#333,stroke-width:2px
```

**Selection Criteria:**
- **Fixed**: Predictable costs, simple projects
- **Dynamic**: Long-running projects, performance matters
- **Priority**: Business-critical work, clear priorities
- **Adaptive**: Experimental projects, uncertain workload

---

## 9. Security Architecture

### 9.1 Defense in Depth

Security implemented across four layers with multiple controls:

```mermaid
graph TB
    subgraph Input["INPUT VALIDATION LAYER"]
        I1["Prompt Injection Detection<br/>──────────────────────<br/>Flags patterns:<br/>• 'Ignore previous instructions'<br/>• 'You are now...'<br/>• 'Print your system prompt'<br/>• 'Disregard all...'<br/>──────────────────────<br/>Action: Warn user, require confirmation"]
        
        I2["Parameter Validation<br/>──────────────────────<br/>Checks:<br/>• Type correctness<br/>• Range boundaries<br/>• Path sanitization<br/>• No directory traversal (../../)<br/>──────────────────────<br/>Action: Reject invalid parameters"]
        
        I3["Command Injection Prevention<br/>────────────────────────────<br/>Enforcement:<br/>• Parameterized execution only<br/>• No string concatenation<br/>• Block shell metacharacters<br/>• Whitelist allowed commands<br/>────────────────────────────<br/>Action: Sanitize or reject"]
    end
    
    subgraph Execution["EXECUTION ISOLATION LAYER"]
        E1["Code Execution Sandbox<br/>────────────────────<br/>Process-level defense in depth:<br/>• No shell; argv[0] allowlist<br/>• Network-egress argument checks<br/>• CPU time limit: 30s<br/>• Memory limit: 512 MB<br/>• ALLOWLIST child env (no keys)<br/>• Dead proxies when network off<br/>────────────────────<br/>HONEST BOUNDARY: not a container;<br/>the eval host enforces real isolation"]
        
        E2["Tool Permission Gating<br/>──────────────────────<br/>Control:<br/>• Tier-based access matrix<br/>• Low-tier models restricted<br/>• Dangerous tools need Tier 3<br/>• Temporary escalation logged<br/>──────────────────────<br/>Action: Deny or grant with logging"]
    end
    
    subgraph Data["DATA PROTECTION LAYER"]
        D1["Secret Detection<br/>────────────────<br/>Pre-commit scan for:<br/>• API keys (long alphanumeric)<br/>• Private keys (BEGIN markers)<br/>• Passwords (password=, pwd=)<br/>• Connection strings<br/>• JWT secrets<br/>────────────────<br/>Action: Block commit, alert specialist"]
        
        D2["File-Level Permissions<br/>──────────────────────<br/>Sensitive files:<br/>• .env files → Read-only<br/>• secrets/* → Blocked entirely<br/>• src/auth/* → Security review required<br/>• migrations/* → Specialist approval needed<br/>──────────────────────<br/>Action: Enforce access rules"]
        
        D3["API Key Management<br/>──────────────────<br/>Best practices:<br/>• Environment variables only<br/>• Never in config files<br/>• Rotation support<br/>• Validation on startup<br/>• Revocation on exposure<br/>──────────────────<br/>Action: Secure storage, validation"]
    end
    
    subgraph Access["ACCESS CONTROL LAYER"]
        A1["Branch Protection<br/>────────────────<br/>Rules:<br/>• main: Architect only<br/>• No force-push to main<br/>• PR required for all merges<br/>• CI/CD must pass<br/>• Minimum 1 reviewer<br/>────────────────<br/>Action: Enforce via GitHub settings"]
        
        A2["Audit Trail<br/>────────────<br/>Logging:<br/>• All actions logged<br/>• Immutable append-only<br/>• Cryptographic signatures<br/>• Tamper detection<br/>• Export for compliance<br/>────────────<br/>Action: Persistent logging"]
    end
    
    Input --> Execution
    Execution --> Data
    Data --> Access
    
    classDef default stroke:#333,stroke-width:2px
```

**Security Principles:**
1. **Defense in Depth**: Multiple independent security layers
2. **Least Privilege**: Minimum necessary permissions
3. **Fail Secure**: Deny by default, grant explicitly
4. **Audit Everything**: Complete action trail for forensics

---

## 10. Verification Pipeline

### 10.1 Five-Stage Quality Gate

Progressive verification ensures correctness before integration:

```mermaid
flowchart LR
    Start([Code<br/>Complete]) --> S1
    
    S1["STAGE 1<br/>Agent Self-Check<br/>─────────────<br/>• Syntax validation<br/>• Local test execution<br/>• Lint verification<br/>• Code formatting<br/>• Acceptance criteria review<br/>─────────────<br/>Gate: Before PR creation"]
    
    S2["STAGE 2<br/>Automated CI/CD<br/>─────────────<br/>• Build verification<br/>• Full test suite<br/>• Code coverage >80%<br/>• Security scanning<br/>• License compliance<br/>─────────────<br/>Gate: Blocks PR merge"]
    
    S3["STAGE 3<br/>Reviewer Agent<br/>─────────────<br/>• Code smell detection<br/>• Pattern consistency<br/>• Documentation check<br/>• Test adequacy<br/>• Performance review<br/>─────────────<br/>Gate: Creates issues (non-blocking)"]
    
    S4["STAGE 4<br/>Manager Review<br/>─────────────<br/>• Acceptance validation<br/>• Conflict detection<br/>• Merge safety check<br/>• Team coordination<br/>─────────────<br/>Gate: Approves for architect"]
    
    S5["STAGE 5<br/>Architect Review<br/>─────────────<br/>• Intent alignment<br/>• Global rules compliance<br/>• Architecture fit<br/>• Quality assessment<br/>─────────────<br/>Gate: Final merge decision"]
    
    Success([Merged to<br/>Main Branch])
    
    S1 --> S2
    S2 --> S3
    S3 --> S4
    S4 --> S5
    S5 --> Success
    
    S1 -.->|Fails| Fix1[Fix & Retry]
    S2 -.->|Fails| Fix2[Fix & Retry]
    S4 -.->|Conflicts| Resolve[Coordinate<br/>Resolution]
    S5 -.->|Rejected| Redesign[Redesign<br/>Approach]
    
    Fix1 -.-> S1
    Fix2 -.-> S2
    Resolve -.-> S4
    Redesign -.-> S1
    
    classDef default stroke:#333,stroke-width:2px
```

**Verification Criteria:**

| Stage | Blocking | Retry Allowed | Typical Duration |
|-------|----------|---------------|------------------|
| 1. Self-Check | Yes | Unlimited | 2-5 minutes |
| 2. CI/CD | Yes | Unlimited | 5-10 minutes |
| 3. Code Review | No | N/A | 1-2 minutes |
| 4. Manager | Yes | Limited (2x) | 1-3 minutes |
| 5. Architect | Yes | Limited (1x) | 2-5 minutes |

**Quality Metrics Tracked:**
- Test pass rate (target: 100%)
- Code coverage (target: >80%)
- Security vulnerabilities (target: 0 critical)
- Linting errors (target: 0)
- Average time to merge (optimization metric)

---

## System Statistics

| Component | Count | Description |
|-----------|-------|-------------|
| **Agents** |
| Architect | 1 | Tier 4 model (GPT-4, Claude Opus) |
| Managers | 3 | Backend, Frontend, DevOps teams |
| Specialists | 10+ | Configurable based on budget and needs |
| **Tools** |
| Tier 1 | 5 | Basic read operations |
| Tier 2 | 7 | Development operations |
| Tier 3 | 8 | Advanced operations |
| **Architecture** |
| Layers | 5 | UI, Orchestration, Coordination, Execution, Infrastructure |
| Verification Stages | 5 | Progressive quality gates |
| Error Recovery Levels | 4 | Graduated escalation hierarchy |
| Budget Strategies | 4 | Selectable allocation approaches |

---

## Key Design Principles

1. **Hierarchical Orchestration**: Clear separation of strategic (Architect), tactical (Manager), and operational (Specialist) responsibilities

2. **Graduated Recovery**: Error handling progresses from autonomous to human-involved, minimizing unnecessary escalation

3. **Evidence-Based Assignment**: Multi-factor algorithm considers expertise, availability, load, and capability for optimal task routing

4. **Progressive Verification**: Five-stage pipeline ensures correctness without unnecessary overhead

5. **Secure by Default**: Defense-in-depth security with input validation, execution isolation, data protection, and access control

6. **Resource Efficiency**: Multiple budget strategies with monitoring and optimization recommendations

7. **Context Efficiency**: Time-windowed compression balances detail retention with memory constraints

8. **Branch Isolation**: Parallel development without conflicts through branch-per-agent strategy

---

## Hackathon Alignment

**Correctness**: Five-stage verification pipeline with automated testing, code review, and multiple approval gates

**Orchestration**: Three-tier hierarchy (Architect → Managers → Specialists) with intelligent task routing

**Recovery**: Four-level error handling with self-recovery, manager intervention, architect escalation, and human fallback

**Efficiency**: Dynamic budget allocation, context compression, tool tier optimization, and performance monitoring

**Autonomy**: Minimal human intervention required, self-recovering agents, automated workflows, and progressive verification

---

**Document Version**: 1.0  
**Created**: September 2026  
**Repository**: https://github.com/MRiARC/ai-coding-harness  
**Design Specification**: See [DESIGN_SPEC.md](DESIGN_SPEC.md) for complete 5,294-line implementation details

## Security boundary (honest statement)

The harness executes commands and test suites from arbitrary repositories. Its
process-level sandbox (`harness/security/sandbox.py`) is **defense in depth,
not a container**:

- **Environment allowlist**: child processes (test runs, code execution)
  receive only PATH/HOME/LANG/locale variables — never the harness's API keys,
  gateway URLs, or tokens. The common leak (a target-repo test printing
  `os.environ`) sees nothing.
- **Dead proxies**: when `tools.allow_network_commands: false`, proxy-aware
  clients point at a closed local port — accidental egress fails fast.
- **Argument-level egress checks**: `git clone`, `pip install`, `npm install`,
  and fetchers (`curl`, `wget`, `ssh`, ...) anywhere in argv are rejected —
  the argv[0] allowlist alone misses them.
- **Resource limits**: 30s CPU, 512 MB memory, project-dir confinement, no
  shell, output truncation.

What this does **not** stop: a determined payload that crafts its own
environment or spawns escape processes. The deployment host's isolation
(container/VM/namespace) is the real boundary for that class — the harness
sandbox exists so the ordinary case never leaks credentials or phones home.
