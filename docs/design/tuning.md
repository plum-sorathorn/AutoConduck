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

With `δ = 1-p`, gamma is `1 + 1.5δ`, biases are `-.20-.15δ` and
`.20-.15δ`, and phase bands shift down by respectively `.15δ`, `.30δ`, and
`.20δ` (minimum width `.05`, lower bound `.02`). Ambiguity expands to
`(.55-.05δ, .70+.05δ)`. EMA alpha is `.10+.10δ` and the quality floor is
`.5-.05δ`. Single-model pools retain pool-relative defaults and receive no
override.

Projected spend is explicitly an estimate: demand and future mix are not
observable, so the tool reports an open-loop caveat. Stats seed request shares
and month-to-date pace when available. A profile is stored separately in
`~/.autoconduck/tune_profile.json`; config edits are backed up first.

For example, `$87` with `25%` headroom gives a `$65.25` target. The resulting
pressure and controls depend on the four configured model prices; the tool
prints the per-model breakdown and warns if the target is unreachable.

Deferred from v1: closed-loop re-pacing, subscription auto-detection, and
multiple named profiles.
