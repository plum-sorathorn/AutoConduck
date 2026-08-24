# Project: AutoConduck 0.3.2 Modernization

## Architecture
AutoConduck 0.3.2 transforms from a heuristic 10-factor regex complexity scorer and static DAG pipeline into an intelligent Dynamic SLM Orchestration Engine.

### Data Flow & Execution Pipeline
1. **Client Request**: `/v1/chat/completions` (OpenAI) or `/v1/messages` (Anthropic).
2. **Turn Guard (`turn_guard.py`)**: Synchronous 0ms classifier. If active tool loop without stagnation, directly relays to active model tier (<2ms overhead). If stagnation (3+ identical tool calls or 2+ consecutive errors) or new user turn, routes to SLM Planner.
3. **Session Guard (`session_guard.py`)**: Enforces immutable prefix contract for upstream prompt caching across 40+ turns; applies 80% context window compaction summarizing tool outputs while preserving code fences and headers.
4. **SLM Architect (`slm_planner.py`)**: Embedded Qwen 2.5 Coder 0.5B Instruct (Q4_K_M GGUF) with Outlines BNF logit-constrained grammar generating deterministic `ExecutionPlan` JSON within <=75ms, guarded by a 100ms circuit breaker failing soft to the balanced tier.
5. **Model Pool (`model_pool.py`)**: Dynamic pool-relative quantile tiering (`cheap_fast`, `balanced`, `frontier_reasoning`) adapting seamlessly to any user-selected pool size (1 to 20+ models) with context ceiling and tool calling filters.
6. **Dynamic LangGraph Factory (`dynamic_factory.py`)**: Transient `StateGraph` compilation with parallel subtask fan-out, conditional LanceDB RAG node, terminal Synthesizer node on `frontier_reasoning`, and `SqliteSaver` session checkpointing.
7. **Dynamic SSE Streamer (`sse_streamer.py`)**: Real-time visual DAG execution state transitions (`[..]`, `[>>]`, `[OK]`, `[ERR]`) emitted as `delta.reasoning_content` (OpenAI) and `thinking_delta` (Anthropic), transitioning smoothly into response markdown.
8. **Selective Knowledge / RAG (`knowledge/`)**: LanceDB embedded vector index for repository dependencies and API contracts (max 250 tokens) injected into `State["verified_context"]`.

---

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| F1 | Version bump to 0.3.0-dev | Update version string in `__init__.py` and `pyproject.toml` | M1 | ORIGINAL_REQUEST §R1 |
| F2 | Runtime Dependency Synchronization | Add `llama-cpp-python`, `outlines`, `lancedb`, `langgraph-checkpoint-sqlite`; remove `semantic-router` | M1 | ORIGINAL_REQUEST §R1 |
| F3 | Compatibility Shims Layer (`_compat/`) | Safe fallback shims for optional binary dependencies | M1 | ORIGINAL_REQUEST §R1 |
| F4 | Turn Guard (`turn_guard.py`) | 0ms classifier for tool-loop bypass (<2ms) and stagnation escalation (3 identical / 2 errors) | M2 | ORIGINAL_REQUEST §R2 |
| F5 | SLM Architect (`slm_planner.py`) | Qwen 2.5 Coder 0.5B Instruct + Outlines BNF grammar + 100ms circuit breaker | M2 | ORIGINAL_REQUEST §R2 |
| F6 | Dynamic LangGraph Factory (`dynamic_factory.py`) | Transient `StateGraph` DAG compilation + parallel fan-out + `SqliteSaver` | M2 | ORIGINAL_REQUEST §R2 |
| F7 | Dynamic SSE Streamer (`sse_streamer.py`) | DAG visual state transitions (`⏳`, `🟢`, `🔴`) as `delta.reasoning_content` and `thinking_delta` | M2 | ORIGINAL_REQUEST §R2 |
| F8 | Selective Knowledge / RAG (`knowledge/`) | LanceDB vector store + max 250 token context in `State["verified_context"]` | M2 | ORIGINAL_REQUEST §R2 |
| F9 | Session Guard (`session_guard.py`) | Immutable prefix prompt caching + 80% context compaction preserving code fences | M2 | ORIGINAL_REQUEST §R2 |
| F10 | Autonomous Model Pool (`model_pool.py`) | 3-tier auto-tiering (`cheap_fast`, `balanced`, `frontier_reasoning`) + context/tool filters | M2 | ORIGINAL_REQUEST §R2 |
| F11 | Server & Dispatcher Integration Wiring | Wire Turn Guard, SLM Planner, Dynamic Factory, SSE Streamer, Session Guard into `server_routes.py` and `dispatcher.py` | M2 | ORIGINAL_REQUEST §R2 |
| F12 | Retained Systems Migration | Adapt `pricing.py` (`select_for_tier`), `stats.py` (`ExecutionPlan` logging), `config.py`, `resolver.py`, `tui/app.py`, `tuning.py`, `digest.py`, `__init__.py` | M3 | ORIGINAL_REQUEST §R3 |
| F13 | Obsolete Test Deletion | Delete `test_routing_fast_path.py`, `test_complexity_and_tuning.py`, `test_empirical_tuning.py` | M3 | ORIGINAL_REQUEST §R3 |
| F14 | Comprehensive Unit & Integration Test Suites | Implement 7 new test suites (`test_turn_guard.py`, `test_slm_planner.py`, `test_dynamic_factory.py`, `test_sse_streamer.py`, `test_rag_node.py`, `test_session_guard.py`, `test_model_pool.py`) and rewrite 4 suites | M3 | ORIGINAL_REQUEST §R3 |
| F15 | Documentation Updates | Update `README.md`, `AGENTS.md`, `docs/design/dynamic-model-selection.md`, `docs/design/tuning.md`, `docs/CHANGELOG.md` | M4 | ORIGINAL_REQUEST §R4 |
| F16 | New Design Documentation | Create `slm-architecture.md`, `dynamic-dag.md`, `session-management.md`, `rag-subsystem.md`, `migration/0.2-to-0.3.md` | M4 | ORIGINAL_REQUEST §R4 |
| F17 | Stale Codebase Cleanup | Delete 11 stale files (`complexity.py`, `evaluator.py`, `semantic_router.py`, `fast_graph.py`, `graph.py`, `compactor.py`, `recon.py`, `progress.py`, `_compat/`, etc.) | M4 | ORIGINAL_REQUEST §R4 |
| F18 | Knowledge Graph Rebuild & Final 0.3.0 Release | Run `graphify update .`, verify 100% test pass rate, bump version to final 0.3.0 | M4 | ORIGINAL_REQUEST §R4 |

---

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Phase 0 — Pre-Flight & Dependency Synchronization | F1, F2, F3: Version bump to 0.3.0-dev, dependencies update, _compat/ shims | none | DONE |
| M2 | Phase 1 & 2 — Green-Field Foundations & Integration Wiring | F4, F5, F6, F7, F8, F9, F10, F11: Green-field foundation modules & pipeline wiring | M1 | IN_PROGRESS |
| M3 | Phase 3 & 4 — Migration of Retained Systems & Test Suite Overhaul | F12, F13, F14: Retained systems adaptation, obsolete test deletion, 7 new test suites, 4 rewrites | M2 | PLANNED |
| M4 | Phase 5, 6 & 7 — Documentation, Stale Cleanup & Validation | F15, F16, F17, F18: Design docs, stale file deletion, graphify update, smoke test, final 0.3.0 release | M3 | PLANNED |

---

## Interface Contracts

### 1. Turn Guard (`turn_guard.py`)
```python
from enum import Enum
from pydantic import BaseModel
from typing import Any

class TurnAction(str, Enum):
    DIRECT_ACTIVE_TIER = "direct_active_tier"
    SLM_PLAN = "slm_plan"
    ESCALATE_SLM = "escalate_slm"

class TurnClassificationResult(BaseModel):
    is_tool_loop: bool
    is_stagnant: bool
    stagnation_reason: str | None = None
    target_action: TurnAction
    tool_call_streak: int = 0
    error_streak: int = 0
    last_tool_name: str | None = None

class TurnGuard:
    def classify_turn(self, messages: list[dict[str, Any]]) -> TurnClassificationResult: ...
```

### 2. SLM Architect (`slm_planner.py`)
```python
from enum import Enum
from pydantic import BaseModel, Field
from typing import Literal

class ModelTier(str, Enum):
    CHEAP_FAST = "cheap_fast"
    BALANCED = "balanced"
    FRONTIER_REASONING = "frontier_reasoning"

class SubTaskSpec(BaseModel):
    id: str
    goal: str
    scope: list[str] = Field(default_factory=list)
    role: Literal["recon", "read", "edit", "verify", "bash", "reasoning"] = "read"
    depends_on: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    output_contract: str = ""
    read_budget: int = 5

class ExecutionPlan(BaseModel):
    route: Literal["fast_direct", "dynamic_dag"]
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    task_type: Literal["chat", "explain", "recon", "single_edit", "multi_edit", "debug", "refactor", "full_workflow"] = "chat"
    suggested_tier: ModelTier = ModelTier.BALANCED
    needs_rag: bool = False
    rag_queries: list[str] = Field(default_factory=list)
    subtasks: list[SubTaskSpec] = Field(default_factory=list)
    synthesizer_tier: ModelTier = ModelTier.FRONTIER_REASONING
    rationale: str = ""
    fallback_used: bool = False

class SLMPlanner:
    async def plan(self, messages: list[dict[str, Any]], config: Any = None) -> ExecutionPlan: ...
```

### 3. Dynamic LangGraph Factory (`dynamic_factory.py`)
```python
from pydantic import BaseModel, Field, ConfigDict
from typing import Any

class DynamicState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    messages: list[dict[str, Any]] = Field(default_factory=list)
    session_id: str = "default"
    thread_id: str = "default"
    plan: ExecutionPlan | None = None
    verified_context: list[str] = Field(default_factory=list)
    subtask_outputs: dict[str, str] = Field(default_factory=dict)
    subtask_errors: dict[str, str] = Field(default_factory=dict)
    active_node: str = "init"
    synthesizer_output: str | None = None
    final_result: Any = None
    is_fallback: bool = False

def build_dynamic_graph(plan: ExecutionPlan, checkpointer: Any = None) -> Any: ...
```

### 4. Dynamic SSE Thinking Streamer (`sse_streamer.py`)
```python
from typing import Literal, AsyncIterator

class SSEThinkingStreamer:
    def __init__(self, client_protocol: Literal["openai", "anthropic"], model_id: str): ...
    async def emit_node_transition(self, node_name: str, status: Literal["pending", "running", "completed", "failed"], detail: str = "") -> str: ...
    async def stream_synthesizer_tokens(self, token_async_iter: AsyncIterator[str]) -> AsyncIterator[str]: ...
```

### 5. Session Guard (`session_guard.py`)
```python
class SessionGuardResult(BaseModel):
    messages: list[dict[str, Any]]
    compacted: bool
    original_tokens: int
    final_tokens: int
    cache_prefix_preserved: bool

class SessionGuard:
    def guard_context(self, messages: list[dict[str, Any]], context_window: int) -> SessionGuardResult: ...
```

### 6. Autonomous Model Pool (`model_pool.py`)
```python
class ModelPool:
    def select_for_tier(self, tier: ModelTier, min_context_window: int = 0, requires_tools: bool = False, pseudo_model: str = "autoconduck") -> str: ...
```

---

## Code Layout
```
autoconduck/
├── __init__.py                     # Package version & top-level exports
├── config.py                       # SelectionConfig & app configuration
├── stats.py                        # ExecutionPlan logging & metrics
├── tuning.py                       # Tuning & budget estimation
├── digest.py                       # Fast-path deterministic context reader
├── resolver.py                     # Route resolver
├── jsonutil.py                     # Safe JSON extraction
├── main.py                         # CLI & supervisor daemon entrypoints
├── update.py                       # Update check
├── _compat/                        # Temporary fallback shims (M1-M3, deleted in M4)
│   ├── __init__.py
│   ├── llama_fallback.py
│   ├── outlines_fallback.py
│   ├── lancedb_fallback.py
│   └── sqlite_checkpointer.py
├── routing/
│   ├── __init__.py
│   ├── dispatcher.py               # Dispatcher routing pipeline
│   ├── pricing.py                  # Cost calculation & select_for_tier
│   ├── slm_planner.py              # Qwen 2.5 Coder 0.5B Instruct + Outlines BNF grammar
│   └── model_pool.py               # 3-tier autonomous model pool
├── orchestrator/
│   ├── __init__.py
│   ├── dynamic_factory.py          # Dynamic LangGraph DAG compiler
│   ├── session_guard.py            # Immutable prefix & 80% compaction guard
│   ├── roles.py                    # Role definitions
│   ├── skeletons.py                # AST skeleton analysis
│   ├── subagents.py                # Subagent execution logic
│   ├── executor_loop.py            # Tool executor loop
│   ├── tools.py                    # Tool definitions & claim registry
│   ├── handoff.py                  # Execution handoff formatting
│   └── helpers.py                  # String and model helpers
├── server/
│   ├── __init__.py
│   ├── turn_guard.py               # 0ms tool loop classifier & stagnation escalation
│   ├── sse_streamer.py             # Live DAG transition thinking streamer
│   ├── server_routes.py            # FastAPI route handlers
│   ├── server_streaming.py         # Streaming lifecycle & disconnect detection
│   ├── messages_api.py             # Anthropic messages translation shim
│   ├── messages_models.py          # Anthropic schema models
│   └── messages_sse.py             # Anthropic SSE framing
├── knowledge/
│   ├── __init__.py
│   ├── vector_store.py             # LanceDB vector store wrapper
│   ├── extractor.py                # AST & manifest extractor
│   └── models.py                   # RAG data models
├── harnesses/                      # 8 Agent harness adapters (Claude, OpenCode, Pi, Aider, etc.)
├── auth/                           # Provider credentials & auth
├── launcher/                       # Daemon launcher & binary PATH shims
├── cli/                            # CLI commands
├── presets/                        # Preset catalog data
└── tui/                            # Textual terminal UI dashboard
```
