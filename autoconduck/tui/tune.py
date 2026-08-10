"""Budget tuning screens.  Calculation and persistence stay testable offline."""
from __future__ import annotations
from datetime import datetime
from .onboarding import _TEXTUAL, _require_textual
from autoconduck.config import get_config, save_config, backup_config
from autoconduck.tuning import (SimpleInputs, compute_tuning, save_profile,
                                load_profile, project_spend, _defaults)

FIELDS = ("monthly_limit", "unit", "headroom_pct", "active_hours_per_month",
          "expected_requests_per_month", "input_output_ratio", "burst_factor")
RANGES = {"gamma": (0.1, 5), "alpha": (0.000001, 1), "quality": (0.000001, 1),
          "bias": (-.5, .5), "burst_factor": (1, 3), "ambiguous": (0, 1)}

def _pool(cfg):
    return [e for e in getattr(cfg, "model_list", []) if e.get("enabled", True) is not False]

def _commit(result, inputs):
    cfg = get_config()
    backup_config()
    selection = cfg.selection
    for name, pair in result.tunables.items():
        if hasattr(selection, name): setattr(selection, name, pair[1])
        elif hasattr(cfg, name): setattr(cfg, name, pair[1])
    for entry in cfg.model_list:
        name = str(entry.get("id") or entry.get("model_name") or entry.get("model") or "")
        if entry.get("enabled", True) is not False and name in result.per_model_limits:
            entry["max_usd_per_min"] = result.per_model_limits[name]
    save_config(cfg)
    save_profile(inputs, result)

if _TEXTUAL:
    from textual.screen import Screen
    from textual.containers import Vertical, Horizontal
    from textual.widgets import Static, Input, Button

    class TuneScreen(Screen):
        OPTIONS = [("Simple Tuning Mode", "Quick budget, headroom, & monthly limit adjustments"),
                   ("Advanced Tuning Mode", "Full parameter, gamma, quality & ambiguity band tuning")]
        def __init__(self, controller=None, mode=None):
            super().__init__(); self.controller = controller; self.mode = mode; self.cursor = 0
        def compose(self):
            yield Vertical(
                Static("┌─ AutoConduck · Budget & Parameter Tuning ─┐"),
                Static(self._render_menu(), id="body", markup=True),
                Static("↑/↓: move · Enter: select mode · [s] Simple · [a] Advanced · [ctrl+c] quit", id="help")
            )
        def _render_menu(self):
            lines = ["Select tuning mode:\n"]
            for i, (title, desc) in enumerate(self.OPTIONS):
                mark = "› " if i == self.cursor else "  "
                text = f"{mark}[bold]{title}[/bold] — {desc}"
                lines.append(f"[reverse]{text}[/reverse]" if i == self.cursor else text)
            return "\n".join(lines)
        def _choose(self):
            if self.controller:
                screen = SimpleTuneScreen(self.controller) if self.cursor == 0 else AdvancedTuneScreen(self.controller)
                self.controller.push_screen(screen)
        def on_key(self, event):
            if event.key == "s":
                if self.controller: self.controller.push_screen(SimpleTuneScreen(self.controller))
            elif event.key == "a":
                if self.controller: self.controller.push_screen(AdvancedTuneScreen(self.controller))
            elif event.key == "down":
                self.cursor = (self.cursor + 1) % len(self.OPTIONS)
                self.query_one("#body", Static).update(self._render_menu())
            elif event.key == "up":
                self.cursor = (self.cursor - 1) % len(self.OPTIONS)
                self.query_one("#body", Static).update(self._render_menu())
            elif event.key == "enter":
                self._choose()

    class SimpleTuneScreen(Screen):
        def __init__(self, controller=None):
            super().__init__(); self.controller = controller; self.expanded = False
            profile = load_profile() or {}; self.values = dict(profile.get("inputs", {}))
        def compose(self):
            v = self.values
            yield Vertical(Static("Simple budget tuning (Tab reveals discretionary inputs)"),
                Input(value=str(v.get("monthly_limit", 87)), placeholder="monthly limit", id="monthly_limit"),
                Input(value=str(v.get("unit", "usd")), placeholder="usd or tokens", id="unit"),
                Input(value=str(v.get("headroom_pct", 25)), placeholder="headroom %", id="headroom_pct"),
                Static("", id="optional"), Static("Enter: preview · Tab: optional · b: back", id="help"))
        def _render_optional(self):
            body = self.query_one("#optional", Static)
            body.update("active_hours_per_month / expected_requests_per_month / input_output_ratio / burst_factor\n"
                        "Edit these values in the Advanced inputs when revealed." if self.expanded else "")
        def _inputs(self):
            def number(key, default, cast=float):
                raw = self.query_one("#" + key, Input).value.strip()
                return cast(raw) if raw else default
            unit = self.query_one("#unit", Input).value.strip().lower()
            if unit not in ("usd", "tokens"): raise ValueError("unit must be usd or tokens")
            return SimpleInputs(number("monthly_limit", 87), unit, number("headroom_pct", 25),
                self.values.get("active_hours_per_month", 160), self.values.get("expected_requests_per_month"),
                self.values.get("input_output_ratio", 3), self.values.get("burst_factor", 1.8))
        def on_key(self, event):
            if event.key == "tab": self.expanded = not self.expanded; self._render_optional()
            elif event.key == "enter":
                try:
                    inputs = self._inputs(); result = compute_tuning(inputs, _pool(get_config()))
                    self.controller.push_screen(TunePreviewScreen(self.controller, inputs, result))
                except (ValueError, TypeError) as exc: self.query_one("#help", Static).update(str(exc))
            elif event.key == "b": self.controller.pop_screen()

    class AdvancedTuneScreen(Screen):
        def __init__(self, controller=None):
            super().__init__(); self.controller = controller; self.cursor = 0
            cfg = get_config(); self.current = {**_defaults(), "ambiguous_low": cfg.ambiguous_low,
                "ambiguous_high": cfg.ambiguous_high, "burst_factor": 1.8}
            self.new = dict(self.current)
        def compose(self): yield Vertical(Static("Tunable | default | range | current | new"), Static(self._table(), id="table"), Static("↑/↓ select · Enter edit · Enter preview · b back", id="help"))
        def _table(self):
            return "\n".join(f"{k:32} {v!s:12} {self._range(k):14} {self.current.get(k)!s:12} {self.new.get(k)!s}" for k,v in self.current.items())
        def _range(self, key):
            if "bias" in key: return "[-.5,.5]"
            if "gamma" in key: return "[.1,5]"
            if "alpha" in key or "quality" in key: return "(0,1]"
            if "ambiguous" in key: return "[0,1]"
            return "non-negative"
        def validate(self, key, value):
            value = float(value)
            if "bias" in key and not -.5 <= value <= .5: raise ValueError("bias must be within ±0.5")
            if "gamma" in key and not .1 <= value <= 5: raise ValueError("gamma must be 0.1–5")
            if ("alpha" in key or "quality" in key) and not 0 < value <= 1: raise ValueError("value must be in (0,1]")
            if "ambiguous" in key and not 0 <= value <= 1: raise ValueError("ambiguous values must be in [0,1]")
            if key == "burst_factor" and not 1 <= value <= 3: raise ValueError("burst factor must be 1–3")
            if "spend_guard" in key and value < 0: raise ValueError("spend guards must be non-negative")
            return value
        def on_key(self, event):
            keys = list(self.current)
            if event.key in ("up", "down"): self.cursor = max(0, min(len(keys)-1, self.cursor + (1 if event.key == "down" else -1)))
            elif event.key == "enter":
                if self.cursor == len(keys)-1: self.controller.push_screen(TunePreviewScreen(self.controller, SimpleInputs(87), compute_tuning(SimpleInputs(87), _pool(get_config()), current=self.new)))
            elif event.key == "b": self.controller.pop_screen()
            self.query_one("#table", Static).update(self._table())

    class TunePreviewScreen(Screen):
        def __init__(self, controller, inputs, result):
            super().__init__(); self.controller=controller; self.inputs=inputs; self.result=result
        def compose(self):
            lines=["Tunable | Current | Proposed"]+[f"{k} | {a} | {b}" for k,(a,b) in self.result.tunables.items()]
            lines += ["", "Warnings: " + ("; ".join(self.result.warnings) or "none"),
                      "EMA caveat: before 3 real requests per model, effective value = raw price ÷ quality_score only.",
                      "c: commit · r: reset defaults · b: back"]
            yield Static("\n".join(lines))
        def on_key(self, event):
            if event.key == "c": _commit(self.result, self.inputs); self.controller.pop_screen()
            elif event.key == "r":
                self.result = compute_tuning(self.inputs, _pool(get_config()), current=_defaults()); self.refresh()
            elif event.key == "b": self.controller.pop_screen()
else:
    class TuneScreen:
        def __init__(self, *args, **kwargs): _require_textual()
    SimpleTuneScreen = AdvancedTuneScreen = TunePreviewScreen = TuneScreen
