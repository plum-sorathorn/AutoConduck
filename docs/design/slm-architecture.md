# Embedded SLM Architecture (Qwen 2.5 Coder 0.5B Instruct)

## 1. Overview
In AutoConduck 0.3.0, the core routing and planning intelligence is powered by an embedded Small Language Model (SLM): **Qwen 2.5 Coder 0.5B Instruct** (quantized via Q4_K_M GGUF). This eliminates static regex-based heuristic complexity scoring and replaces it with structured cognitive task planning under 100ms.

## 2. Invariants & Performance Constraints
- **Sub-100ms Inference**: Quantized local weights execute on CPU/GPU via llama-cpp-python or native shims.
- **Strict JSON Schema Conformance**: Guided generation via Outlines/BNF grammars or Pydantic JSON schema constraints.
- **Circuit Breaker**: Any SLM timeout (>100ms) or parsing error trips the circuit breaker and falls back immediately to deterministic fast-path tier selection without crashing.

## 3. Schema & Output Contract
The SLM produces an ExecutionPlan with the following structure:
- 
oute: ast_direct | dynamic_dag
- confidence: float [0.0, 1.0]
- 	ask_type: chat | xplain | 
econ | single_edit | multi_edit | debug | 
efactor | ull_workflow
- suggested_tier: cheap_fast | alanced | rontier_reasoning
- 
eeds_rag: boolean flag
- 
ag_queries: list of query strings
- subtasks: list of SubTaskSpec items
- synthesizer_tier: target tier for final synthesis
