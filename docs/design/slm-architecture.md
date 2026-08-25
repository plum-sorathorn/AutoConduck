# Embedded SLM Architecture (Qwen 2.5 Coder 0.5B Instruct)

## 1. Overview
In AutoConduck 0.3.4, the core routing and planning intelligence is powered by an embedded Small Language Model (SLM): **Qwen 2.5 Coder 0.5B Instruct** (quantized via Q4_K_M GGUF). This eliminates static regex-based heuristic complexity scoring and replaces it with structured cognitive task planning under 100ms.

## 2. Invariants & Performance Constraints
- **Sub-100ms Inference**: Quantized local weights execute on CPU/GPU via llama-cpp-python or native shims.
- **Strict JSON Schema Conformance**: Guided generation via Outlines/BNF grammars or Pydantic JSON schema constraints.
- **Circuit Breaker**: Any SLM timeout (>100ms) or parsing error trips the circuit breaker and falls back immediately to deterministic SLA-based direct dispatch without crashing.

## 3. Schema & Output Contract
The SLM produces an `ExecutionPlan` with the following structure:
- `route`: `fast_direct` | `dynamic_dag`
- `confidence`: float [0.0, 1.0]
- `task_type`: `chat` | `explain` | `recon` | `single_edit` | `multi_edit` | `debug` | `refactor` | `full_workflow` | `git_ops` | `routine`
- `suggested_sla`: `CapabilitySLA` requirements for the selected model
- `needs_rag`: boolean flag
- `rag_queries`: list of query strings
- `subtasks`: list of `SubTaskSpec` items
- `synthesizer_sla`: `CapabilitySLA` requirements for final synthesis
