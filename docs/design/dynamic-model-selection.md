# Architectural Overhaul: Model Selection

After analyzing the current routing mechanics (specifically in `routing/model_pool.py` and `routing/pricing.py`) and cross-referencing with the `grok-4.6` runaway cost behavior in your recent logs, I have identified several critical flaws in the existing model selection architecture. 

Below is a ruthless critique of the current system, followed by a proposal for a highly intelligent, metadata-driven approach that eliminates arbitrary tiers, budget thresholds, and retrospective cost scaling.

---

## 1. Critique of the Current Architecture

### 1.1 Logarithmic Cost Scaling masks extreme price differences
The system uses `ln(1 + cost)` to compute distances between models in a given tier. 
* **The Flaw:** Logarithmic scaling severely compresses absolute differences. A model that costs $15.00/M tokens appears only marginally more "distant" than a model costing $2.00/M tokens in log-space.
* **The Symptom:** When `grok-4.6` was evaluated for the `frontier_reasoning` tier, its high absolute cost was masked by the log function, allowing the router to pick it continuously without triggering any rigid cost-ceiling bounds.

### 1.2 EMA Realized-Cost Blending is retrospective
The architecture attempts to blend advertised cost with actual realized cost via Exponential Moving Average (EMA).
* **The Flaw:** EMA learns *after the damage is done*. It’s a trailing indicator. The router only discovers a model is too expensive for a task type *after* it has already burned through tokens (e.g., racking up a whole dollar).
* **The Symptom:** High-speed `dynamic_dag` loops quickly accrued cost because the system needed multiple expensive failures before the EMA "learned" to exclude the model.

### 1.3 Static 3-Tier Segmentation is too rigid
The system forces all available models into three hardcoded buckets: `cheap_fast`, `balanced`, and `frontier_reasoning`.
* **The Flaw:** If a user configures a pool of 20 diverse models, squashing them into 3 buckets destroys resolution. A highly capable $0.80/M model might be shoved into "budget" while a marginally better $1.20/M model lands in "balanced", completely ignoring their specific capabilities.
* **The Symptom:** Tasks are assigned a tier rather than matching task *requirements* to model *capabilities*.

### 1.4 Tuning and Budgets are reactive abstractions
* **The Flaw:** A budget-driven system races to the bottom when the budget runs low, resulting in wildly inconsistent output quality (great early in the month, garbage later). The "tuning" slider attempts to abstract the cost/quality tradeoff but obscures the actual requirements of the code tasks being performed.

---

## 2. Proposal: Metadata-Driven Capability Matching

We should abandon arbitrary tiers, log-scaling, and retrospective EMA. Instead, model selection should function like a **Query Optimizer** matching strict subtask requirements against a multidimensional capability matrix.

### 2.1 The Multidimensional Capability Matrix
Each model is defined by objective metadata, not relative tiers:
* `cost_input` / `cost_output` (Absolute $)
* `context_window` (e.g., 8k, 128k, 2M)
* `supports_vision` (boolean)
* `reasoning_depth` (e.g., none, standard, deep)
* `tool_reliability` (e.g., strong, weak)
* `latency_profile` (e.g., low, high)

### 2.2 Task-Driven SLA (Service Level Agreement)
The SLM (Qwen 2.5 Coder) or predefined heuristics generate an SLA for *every single subtask* in the `ExecutionPlan`.
* **Example (Grep/Read Task):**
  * `min_context`: 64k
  * `tool_reliability`: strong
  * `reasoning_depth`: none
  * `max_cost_output`: $1.00/M
* **Example (Complex Refactor):**
  * `min_context`: 32k
  * `tool_reliability`: strong
  * `reasoning_depth`: deep
  * `max_cost_output`: $15.00/M

### 2.3 Strict Zero-Overhead Filtering (The Query Optimizer)
Selection becomes a simple filter-and-sort operation:
```sql
SELECT model 
FROM user_active_pool 
WHERE context_window >= task.min_context 
  AND reasoning_depth >= task.reasoning_depth
  AND cost_output <= task.max_cost_output
ORDER BY cost_output ASC, latency_profile ASC
LIMIT 1
```
* **Why it works:** If a task strictly requires deep reasoning and a 128k context, it filters out Claude Haiku instantly. It then selects the absolute cheapest model that satisfies those constraints. It never picks `grok-4.6` if a cheaper model meets the SLA. 
* **Zero Surprises:** Cost ceilings are enforced *per-turn* proactively. It is impossible to rack up a dollar on simple reads because the `max_cost_output` for a Read task would strictly prohibit selecting an expensive model, regardless of how much budget remains.

### 2.4 Dropping Budget & Tuning
By matching SLA to Capabilities, the system guarantees the most cost-efficient execution possible for the requested task. 
* **Tuning** is replaced by **Task Criticality** (e.g., a "Draft" mode vs "Production" mode that alters the SLA generation).
* **Budgets** are replaced by **Per-Turn Cost Ceilings** ensuring predictable spend profiles.

