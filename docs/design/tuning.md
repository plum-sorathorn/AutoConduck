# Budget-driven routing tuning

`autoconduck tune` is an open-loop calibration tool. It turns a monthly USD
or token budget and headroom into routing controls using the enabled model
pool. Token budgets use the blended price
`(3 * price_in + price_out) / 4` by default (prices are USD per million
tokens); observed model shares are used once at least three days of data exist.

The target is `monthly_limit * (1 - headroom / 100)`, and the per-minute rate
is target divided by active hours times 60. Pressure uses the pricing module's
log1p scale:

```
p = clamp((log1p(c_max) - log1p(target_position)) /
          (log1p(c_max) - log1p(c_min)), 0, 1)
```

`target_position` maps the rate through the documented 3,000 tokens/minute
assumption. Zero prices use epsilon `.001`; zero-price models are excluded
from the maximum and receive weight 1.0. The guard is `rate * burst_factor`,
with a `.001` floor. Per-model guards are `guard * (.3 + .7 * weight)`, where
`weight = 1 - (log1p(blended) - log1p(c_min))/(log1p(c_max)-log1p(c_min))`.

Under budget pressure `p`, `gamma` is `1 + 2.0p` (curving cost targets steeply towards cheaper models under high pressure), budget bias is `-.20-.20p`, expensive bias is `.20-.35p`, and phase bands shift down by respectively `.20p` (planner), `.20p` (subagent), and `.25p` (executor), with minimum width `.05` and lower bound `.02`. Ambiguity bounds become `(.60+.05p, .75+.05p)`. EMA alpha is `.10+.10p` and the quality floor is `.5`. Single-model pools retain pool-relative defaults and receive no override.

Projected spend is explicitly an estimate: demand and future mix are not
observable, so the tool reports an open-loop caveat. Stats seed request shares
and month-to-date pace when available. A profile is stored separately in
`~/.autoconduck/tune_profile.json`; config edits are backed up first.

For example, `$87` with `25%` headroom gives a `$65.25` target. The resulting
pressure and controls depend on the four configured model prices; the tool
prints the per-model breakdown and warns if the target is unreachable.

Runtime safeguards complement the open-loop profile:

- The realized spend guard uses a 300-second rolling window by default, reducing
  one-request oscillation while preserving per-model `max_usd_per_min` limits.
- Ambiguous prompts only invoke the paid LLM tiebreaker when heuristic complexity
  is at least `0.45`; `autoconduck-budget` raises this floor to `0.65`. The
  tiebreaker itself is opt-in (`tiebreaker_enabled`, default false) —
  lower-value ambiguous work stays on the deterministic fast path, which saves
  both time and the extra classification call.
- The budget pseudo-model still applies its negative closest-cost bias; the
  tiebreaker floor is a latency/cost gate, not a second model-price adjustment.
- Realized request cost is normalized back to USD per million observed tokens
  before it is compared with configured model prices. This prevents EMA warm-up
  from making a model appear artificially cheap because of mixed units.

Deferred from v1: closed-loop re-pacing, subscription auto-detection, and
multiple named profiles. Until closed-loop pacing exists, the simple tuner is a
calibration aid rather than a hard monthly-budget guarantee.
