# Dynamic Model Selection (cost-ratio closest-match) — Redesign Spec

# 1. Dynamic task-value computation

## 1a. Fast path — pure-math V ∈ [0,1]

`evaluator.complexity_of` is evolved (not replaced) into a 7-factor weighted sum, all extracted by single-pass regex/counting over the prompt text — O(n) over a few KB, well under 5ms, zero I/O:

| Factor | Weight | Extraction | Rationale |
| --- | --- | --- | --- |
| `length` | 0.15 | `min(1, char_len/1200)` | scope proxy (down-weighted from 0.20 — weak signal alone) |
| `refs` | 0.10 | `min(1, ref_count/3)`, `ref_count` = regex hits of `@\w+`, `#\d+`, backtick spans, URLs | cross-reference density |
| `structural` | 0.25 | `min(1, structural_count/3)`, count of bullets/numbered lines/code fences/`##` headers | still dominant — multi-part asks are structurally visible (down from 0.50 to make room for new factors) |
| `files` | 0.10 | `min(1, file_count/3)`, regex `[\w\-/\\]+\.\w{1,4}` | breadth of touched surface |
| `keyword_domain` (NEW) | 0.15 | `(clip(hard_hits-easy_hits,-3,3)+3)/6`; hard lexicon: architecture, refactor, migrate, race condition, concurrency, security, distributed, optimize, algorithm; easy lexicon: typo, rename, format, comment, lint, simple | domain-difficulty signal independent of length |
| `edit_intent` (NEW) | 0.15 | regex for imperative edit verbs (fix/implement/add/refactor/write/build) vs read/explain verbs (explain/what/why/describe/review/summarize) → 1.0 edit-only, 0.0 read-only, 0.5 both/neither | read/explain tasks need less firepower than write/edit tasks of similar length |
| `multi_step` (NEW) | 0.10 | `min(1, marker_count/3)`, markers = "then\|next\|after that\|also\|finally" hits + max(0, numbered_items-1) | proxy for sequential complexity |

Weights sum to 1.00, all config-driven. Formula unchanged in shape:

```
complexity = min(1.0, Σ weight_i * factor_i)
if stack_trace_detected: complexity = min(1.0, complexity + stack_trace_boost)  # +0.25, unchanged
```

This IS the fast-path task value V — no separate computation needed; it's already produced by `evaluator.score()` today and simply gets richer. Path routing (fast/slow/ambiguous, hysteresis, ambiguous zone) is **unchanged** — only what happens with the resulting float changes.

## 1b. Slow-path per-phase dynamic targets

Bands (bounds, not points) live in config:

```
phase_bands: {planner: [0.55, 0.85], subagent: [0.10, 0.55], executor: [0.35, 0.70]}
```

**Planner** — target scales linearly with overall task value (no LLM call needed yet, uses the T_i evaluator already computed pre-orchestrator):

```
planner_target(task_value, bands) = lo + (hi - lo) * task_value
```

**Subagent** — combines local subtask complexity (reusing the same pure `complexity_of` extractor on the subtask prompt), an LLM-informed `budget_hint` from `TaskPlan` (deterministic fallback = overall `task_value` when absent/invalid), a role weight, and a breadth-damping term (more parallel subtasks ⇒ each individually simpler slice):

```
role_weight = {"read": 0.3, "analysis": 0.6, "write": 0.9}[role]
breadth_damp = 1 / sqrt(max(1, plan_breadth))
raw = 0.4*complexity_of(subtask_prompt) + 0.3*budget_hint*role_weight + 0.3*breadth_damp
subagent_target = lo + (hi - lo) * clip(raw, 0, 1)
```

**Executor** — overall task value + complexity of the compacted summary (synthesis difficulty proxy) + how many pieces are being integrated:

```
accumulation = min(1.0, subtask_count / 6)
raw = 0.5*task_value + 0.3*complexity_of(compactor_summary) + 0.2*accumulation
executor_target = lo + (hi - lo) * clip(raw, 0, 1)
```

`TaskPlan.budget_hint: float | None = None` — planner LLM is prompted to *optionally* emit it (0.0–1.0); any missing/out-of-range/non-numeric value falls back deterministically to `task_value`. The selection math itself (the formulas above and `select_closest`) is 100% pure — only the *input* `budget_hint` is LLM-informed, never the arithmetic.

## 1c. Role constrains range, not a number

Bands filter the **eligible model set** before distance-matching (not just clamp the target number): `eligible = [m for m in pool if lo <= scaled_cost(m) <= hi] or pool` (fallback to full pool when the band contains zero models — e.g. a 2-model pool). The phase formulas above already produce a target *inside* the band, so normally the nearest model in-band wins; the fallback guarantees a small pool never yields an empty candidate set.

# 2. Closest-model matching (cost ratio)

Both value and cost live in the same `[0,1]` scaled domain already produced by `scaled_cost` (`log1p(price)/max_log1p_in_pool`). **Decision: identity mapping**, target_scaled_cost = value, with a tunable exponent for future calibration:

```python
def target_scaled_cost(value: float, pseudo_model: str | None, config) -> float:
    gamma = config.selection.value_to_cost_gamma          # default 1.0
    bias = 0.0
    if config.selection.pseudo_bias_enabled:
        bias = {"autoconduck-budget": config.selection.pseudo_bias_budget,      # default -0.20
                "autoconduck-expensive": config.selection.pseudo_bias_expensive # default +0.20
               }.get(pseudo_model, 0.0)
    return clip(value ** gamma + bias, 0.0, 1.0)
```

*Alternative considered:* map value to a cost-rank percentile instead — rejected, pool-size-sensitive and unstable as models are added/removed.

**Distance metric & tie-break:** `distance(m) = abs(scaled_cost(m) - target)`; sort candidates by `(distance, scaled_cost, model_id)` → cheaper wins exact ties, then lexical id for determinism/testability. Select `ranked[0]`.

**EMA integration:** after `ema_min_samples` (default 3), normalize the observed
EMA request cost back to the same USD-per-million-token unit as configured
prices, then apply the learned success-rate floor and optional `quality_score`.
If no usable observed token/cost data exists, use `price_in + price_out`.
`scaled_cost()` computes the complete pool's effective values and denominator
once per selection so closest-cost ranking remains linear in pool size.

**Degraded exclusion / all-degraded fallback:**

```python
def select_closest(pool, value, config, *, pseudo_model=None, band=None, degraded=None):
    try:
        eligible = [m for m in pool if not (degraded and m in degraded)]
        if band:
            lo, hi = band
            in_band = [m for m in eligible if lo <= scaled_cost(m, config) <= hi]
            eligible = in_band or eligible
        if not eligible:
            return cheapest_enabled(config)          # all degraded -> ignore degradation, go cheapest overall
        target = target_scaled_cost(value, pseudo_model, config)
        ranked = sorted(eligible, key=lambda m: (abs(scaled_cost(m, config) - target), scaled_cost(m, config), m))
        return ranked[0]
    except Exception:
        return cheapest_enabled(config)               # any error whatsoever -> safe deterministic fallback
```

`cheapest_enabled(config)` is trivial and can't-fail (`min(enabled, key=lambda m: (price_in+price_out, id))`, no EMA, no degraded lookups) — guarantees the fast path never breaks regardless of what goes wrong upstream.

Pseudo-model backward compat: `autoconduck`/`autoconduck-budget`/`autoconduck-expensive` unchanged as names; behavior changes from *hard min/max* to *biased closest-match* — flagged as an explicit behavior change.

# 3. Rewiring call sites

```python
@dataclass
class RoutingDecision:
    path: Literal["fast", "slow", "ambiguous"]
    confidence_band: str
    confidence: float
    complexity: float
    model: str | None = None     # NEW — populated only when path == "fast"; None on slow (phases pick per-role)
```

- **dispatcher.route()**: after resolving `path == "fast"`, compute `decision.model = pricing.select_closest(pool_ids(config), decision.complexity, config, pseudo_model=pseudo_model)` (dispatcher owns pool/config context already — single source of truth, no duplication in resolver).
- **resolver.py**: drop its own `pricing.select(...)` call; just consume `decision.model` (already safe/non-None by construction of `select_closest`).
- **Interactive tool loop fast-path**: Requests containing tool definitions (`tools`), `tool_calls`, or `tool_result` (e.g. from Claude Code, OpenCode, or Pi multi-turn tool steps) are automatically forced to the fast path. This eliminates multi-agent orchestration overhead within CLI agent tool loops, preserving low latency (<5ms routing) and direct tool execution contracts.
- **Ambiguous tiebreaker**: low-value ambiguous work stays deterministic: the
  LLM call is enabled only at complexity `>=0.45`, or `>=0.65` for
  `autoconduck-budget`. Injected test/custom tiebreakers still run explicitly.
  Valid replies are `FAST`/`SLOW` with an optional digit 1-9; digit 0 and trailing
  junk are rejected. If a valid digit is present, use
  `0.5*heuristic_complexity + 0.5*(digit/9)`; otherwise retain heuristic
  complexity. The tiebreak call has a 1.5-second timeout with payload truncation to 500 characters for minimal latency, and failures degrade to fast.
- **planner.py**: `_model_name(cfg)` → `_model_name(cfg, task_value, config)`:

  ```python
  def _model_name(cfg, task_value, config):
      lo, hi = config.selection.phase_bands["planner"]
      target = lo + (hi - lo) * task_value
      return pricing.select_closest(pool_ids(config), target, config, band=(lo, hi))
  ```

  outer try/except keeps `"gpt-4o"` as ultimate fallback (defense-in-depth on top of `select_closest`'s own guard).
- **subagents.py:45**: `qualify_model(select_model_by_tier("cheap", cfg))` → `qualify_model(pricing.select_closest(pool_ids(config), subagent_target(subtask.prompt, subtask.role, plan_breadth, budget_hint, config), config, band=config.selection.phase_bands["subagent"]))`. Requires `role`/`budget_hint`/`plan_breadth` threaded onto whatever fans subagents out today.
- **graph.py:55-66** `_executor_model(pseudo_model, cfg)` → `_executor_model(pseudo_model, cfg, task_value, compactor_summary, subtask_count)`, computing `executor_target(...)` then `pricing.select_closest(..., pseudo_model=pseudo_model, band=cfg.selection.phase_bands["executor"])`; keep `select_model_by_tier("expensive", cfg)` as the outer-except safety net.
- **compactor.py**: unchanged.
- **`select_model_by_tier`**: **kept as a deprecated thin wrapper**, not deleted (backward compat for any external/test callers keying off tier strings):

  ```python
  def select_model_by_tier(tier: str, config) -> str:
      value = {"cheap": 0.15, "budget": 0.15, "mid": 0.5, "balanced": 0.5,
               "reasoning": 0.7, "expensive": 0.85}.get(tier, 0.5)
      return select_closest(pool_ids(config), value, config)
  ```

  `select()` similarly becomes a thin wrapper over `select_closest` with `value` defaulting to 0.15 and pseudo-model bias applied identically.
- **Invariant preserved**: any exception inside `orchestrator/graph.py:run()` (including from these new calls, though they self-guard) is still caught by the existing outer try/except → returns `None` → resolver/dispatcher fall back to the fast path with a `RoutingDecision.model` that is *always* populated (never `None` on the fast path).

# 4. Config & compat surface

New optional top-level block, defaults applied via `field(default_factory=...)` so existing config.yaml files without it keep loading unchanged:

```yaml
selection:
  value_to_cost_gamma: 1.0
  pseudo_bias_budget: -0.20
  pseudo_bias_expensive: 0.20
  pseudo_bias_enabled: true
  ema_min_samples: 3
  closeness_epsilon: 0.02        # used only in tests/logging for tie-detection, not selection branching
  expose_value_in_stats: true
  phase_bands:
    planner:  [0.55, 0.85]
    subagent: [0.10, 0.55]
    executor: [0.35, 0.70]
  complexity_weights:
    length: 0.15
    refs: 0.10
    structural: 0.25
    files: 0.10
    keyword_domain: 0.15
    edit_intent: 0.15
    multi_step: 0.10
```

- `ModelEntry.tier` field: **kept**, becomes advisory/display-only — no selection code reads it anymore except through the deprecated `select_model_by_tier` label→value map. TUI keeps showing the label as a human hint; actual routing is price-driven.
- Pseudo-models `autoconduck`/`autoconduck-budget`/`autoconduck-expensive` unchanged in name and `/v1/models` listing; only internal bias math changes.
- V exposed to users: piggybacked onto the **existing** `/stats` audit surface (no new telemetry layer) — add fields `last_task_value`, `last_target_scaled_cost`, `last_selected_model`, gated by `expose_value_in_stats`.

# 5. Test plan

New files: `tests/test_valuation.py`, `tests/test_pricing_select_closest.py`, `tests/test_phase_bands.py`. Extend existing `tests/test_evaluator.py`, `tests/test_dispatcher.py`, orchestrator test files.

1. `test_complexity_weights_sum_to_one_default` — config default sanity.
2. `test_complexity_of_deterministic_fixed_prompt` — fixed string → pinned exact float (regression anchor).
3. `test_complexity_of_edit_intent_raises_value` — "fix the race condition..." scores higher than "explain what this does".
4. `test_complexity_of_multistep_markers` — "...then...then...finally" > flat single-sentence prompt.
5. `test_select_closest_picks_nearest_not_cheapest` — 5-model pool, value=0.5 → asserts the *middle*-cost model wins, not index 0/-1 (proves fixed-tier behavior is gone).
6. `test_select_closest_pseudo_bias_shifts_target` — same pool, `autoconduck-budget` vs `autoconduck-expensive` select strictly lower/higher-cost models than plain `autoconduck`.
7. `test_select_closest_excludes_degraded` — nearest model is degraded → next-nearest chosen instead.
8. `test_select_closest_all_degraded_falls_back_to_cheapest` — all degraded → global cheapest returned, no exception.
9. `test_select_closest_near_tie_deterministic` — two equidistant models → cheaper wins, and repeated calls are stable.
10. `test_select_closest_never_raises_on_bad_config` — empty/malformed pool → returns `cheapest_enabled` fallback, no throw.
11. `test_phase_band_filters_pool_planner` — band with zero in-band models falls back to full pool; band with 2 in-band models restricts correctly.
12. `test_planner_target_scales_with_task_value` — value 0.1 → target near band low; value 0.9 → near band high.
13. `test_subagent_target_read_role_cheaper_than_write_role` — same subtask text, `role="read"` < `role="write"` target.
14. `test_ambiguous_tiebreak_blends_llm_digit_with_heuristic` — mocked LiteLLM reply "SLOW 7" blends per formula; "SLOW" (no digit) falls back to heuristic only, no exception.
15. `test_orchestrator_error_degrades_to_fast_path_with_model` — force `graph.run()` to raise → resulting decision has `path="fast"` AND `model is not None`.

# 6. Implementation order

1. **`config.py`** — add `SelectionConfig` dataclass + defaults, wire into `AppConfig`, preserve `tier:` YAML parsing.
2. **`routing/evaluator.py`** — config-driven weights, add `keyword_domain`/`edit_intent`/`multi_step` factors, keep `complexity_of(text, config=None)` signature backward-compatible. Run `pytest tests/test_evaluator.py`.
3. **`routing/pricing.py`** — add `target_scaled_cost`, `select_closest`, `cheapest_enabled`, `_entry_effective_value` EMA blend; turn `select`/`select_model_by_tier` into thin deprecated wrappers. Run new `tests/test_pricing_select_closest.py`.
4. **`routing/dispatcher.py`** — add `RoutingDecision.model`, populate on fast path, extend tiebreaker prompt+parsing.
5. **`resolver.py`** — consume `decision.model` instead of calling `pricing.select` directly.
6. **`orchestrator/planner.py`** — add `TaskPlan.budget_hint`, replace `_model_name` with `planner_target` + `select_closest`, update planner system prompt.
7. **`orchestrator/subagents.py`** — thread `role`/`plan_breadth`/`budget_hint` through subagent dispatch, implement `subagent_target`, replace the line-45 `select_model_by_tier("cheap", cfg)` call.
8. **`orchestrator/graph.py`** — replace `_executor_model` (lines 55-66) with `executor_target` + `select_closest`, threading `task_value`/`compactor_summary`/`subtask_count`.
9. **`model_presets.py`** — no functional change; tier labels remain purely advisory (comment only).
10. **`/stats` surface** — add `last_task_value`/`last_target_scaled_cost`/`last_selected_model` fields gated by `expose_value_in_stats`.
11. **`tui/`** — cosmetic pass on any copy implying tier controls routing (P2, non-blocking).
12. New test files per §5, plus updates to existing orchestrator/dispatcher tests replacing tier assertions with value/band assertions.
13. Full `pytest` run; confirm green; remove now-unused internal call sites for `select`/`select_model_by_tier` (keep the functions themselves as the deprecated bridge).

## Value-aware selection (cost per successful task)

Selection learns realized value from local traffic. After `ema_min_samples`, a
model's effective value is its realized dollars per request divided by
`max(success_rate, quality_min_success_rate)` and then divided by its optional
pool-entry `quality_score` (a static prior in `(0, 1]`). This lets known weaker
models be discounted without external benchmark scraping while success rates
are learned from the user's own traffic. `ema_alpha` controls the cost EMA.

The optional spend guard tracks realized spend in a rolling
`spend_guard_window_s` window and excludes models whose annualized-to-minute
rate exceeds `spend_guard_max_usd_per_min`; a pool entry's `max_usd_per_min`
overrides the global cap. If every model is over budget, normal selection is
retained as a safe fallback. The guard can be disabled in selection config.

## API keys and auth file

Provider credentials are stored outside `config.yaml` in
`$AUTOCONDUCK_HOME/auth.yaml` (normally `~/.autoconduck/auth.yaml`):

```yaml
providers:
  openai: sk-...
  my-gateway: env:MY_GATEWAY_KEY
```

Values may be literal keys or `env:NAME` references, resolved when read. Key
precedence is auth file, legacy literal `api_key` in `config.yaml`, then
`api_key_env`. Existing literal keys are migrated automatically on load,
written to the provider entry, removed from config, and backed up first.
On POSIX systems the auth file is created with mode `0600`.
