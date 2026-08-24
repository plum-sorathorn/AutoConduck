# Dynamic DAG LangGraph Factory

## 1. Overview
AutoConduck 0.3.0 replaces fixed 6-phase static orchestrators with an on-the-fly **Dynamic StateGraph Factory** (orchestrator/dynamic_factory.py). Graphs are compiled dynamically at runtime based on the topology of subtasks defined in the SLM ExecutionPlan.

## 2. Dynamic Graph Topology
1. **Init Node (init)**: Initializes execution metadata, session IDs, and loads verified context.
2. **RAG Node (
ag)**: (Optional) Ingests semantic code snippets from LanceDB when plan.needs_rag == True.
3. **Subtask Fan-Out Nodes (subtask_*)**: Generated dynamically per subtask. Independent subtasks execute concurrently in parallel worker threads.
4. **Synthesizer Node (synthesizer)**: Combines verified context and subtask outputs into a cohesive final answer.

## 3. Concurrency & State Reducer Architecture
To support arbitrary parallel fan-out without LangGraph InvalidUpdateError conflicts, DynamicState uses 	yping.Annotated reducers:
- subtask_outputs: Annotated[dict[str, str], _merge_dict]
- subtask_errors: Annotated[dict[str, str], _merge_dict]
- erified_context: Annotated[list[str], operator.add]
- ctive_node: Annotated[str, _latest_val]

## 4. SQLite State Checkpointing
State is persisted via SqliteSaverFallback inheriting from BaseCheckpointSaver, guarded by thread-safe 	hreading.Lock() to prevent multi-threaded SQLite concurrency collisions.
