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
        OPTIONS = [
            ("Simple Tuning Mode", "Quick budget, unit, & monthly limit adjustments"),
            ("Advanced Tuning Mode", "Full parameter, gamma, quality & ambiguity band tuning"),
        ]

        def __init__(self, controller=None, mode=None):
            super().__init__()
            self.controller = controller
            self.mode = mode
            self.cursor = 0

        def compose(self):
            yield Vertical(
                Static("AutoConduck - Budget & Parameter Tuning"),
                Static(self._render_menu(), id="body", markup=True),
                Static(
                    "[up/down] navigate  [enter] select  [s] simple  [a] advanced  [esc/left] back  [ctrl+c] quit",
                    id="help",
                    markup=False,
                ),
            )

        def _render_menu(self):
            lines = ["Select tuning mode:\n"]
            for i, (title, desc) in enumerate(self.OPTIONS):
                mark = ">> " if i == self.cursor else "   "
                text = f"{mark}[bold]{title}[/bold] - {desc}"
                lines.append(f"[reverse]{text}[/reverse]" if i == self.cursor else text)
            return "\n".join(lines)

        def _choose(self):
            if self.controller:
                screen = (
                    SimpleTuneScreen(self.controller)
                    if self.cursor == 0
                    else AdvancedTuneScreen(self.controller)
                )
                self.controller.push_screen(screen)

        def on_key(self, event):
            if event.key == "s":
                if self.controller:
                    self.controller.push_screen(SimpleTuneScreen(self.controller))
            elif event.key == "a":
                if self.controller:
                    self.controller.push_screen(AdvancedTuneScreen(self.controller))
            elif event.key == "down":
                self.cursor = (self.cursor + 1) % len(self.OPTIONS)
                self.query_one("#body", Static).update(self._render_menu())
            elif event.key == "up":
                self.cursor = (self.cursor - 1) % len(self.OPTIONS)
                self.query_one("#body", Static).update(self._render_menu())
            elif event.key == "enter":
                self._choose()
            elif event.key in ("escape", "left", "b"):
                if self.controller:
                    self.controller.pop_screen()

    class SimpleTuneScreen(Screen):
        def __init__(self, controller=None):
            super().__init__()
            self.controller = controller
            profile = load_profile() or {}
            self.values = dict(profile.get("inputs", {}))
            self.unit = str(self.values.get("unit", "usd")).lower()
            self._message = ""

        def _render_unit(self) -> str:
            if self.unit == "usd":
                return "Unit: [bold green][ USD $[/bold green] | [dim]Tokens ][/dim]  (press 'u' or space to toggle)"
            return "Unit: [dim][ USD $[/dim] | [bold cyan]Tokens ][/bold cyan]  (press 'u' or space to toggle)"

        def compose(self):
            v = self.values
            yield Vertical(
                Static("AutoConduck - Simple Budget Tuning"),
                Static(
                    "Set your estimated monthly spend limit to auto-tune cost boundaries."
                ),
                Static(self._render_unit(), id="unit_toggle", markup=True),
                Static("Monthly Budget Limit:"),
                Input(
                    value=str(v.get("monthly_limit", 87)),
                    placeholder="monthly limit (e.g. 50.00)",
                    id="monthly_limit",
                ),
                Static("", id="message", markup=True),
                Static(
                    "[enter] preview & tune  [u/space] toggle unit  [esc] back  [ctrl+c] quit",
                    id="help",
                    markup=False,
                ),
            )

        def _inputs(self):
            raw = self.query_one("#monthly_limit", Input).value.strip()
            try:
                monthly_limit = float(raw) if raw else 87.0
            except ValueError:
                raise ValueError("Monthly limit must be a valid number")
            headroom_pct = float(self.values.get("headroom_pct", 25.0))
            return SimpleInputs(
                monthly_limit,
                self.unit,
                headroom_pct,
                self.values.get("active_hours_per_month", 160),
                self.values.get("expected_requests_per_month"),
                self.values.get("input_output_ratio", 3),
                self.values.get("burst_factor", 1.8),
            )

        def on_key(self, event):
            if event.key in ("u",):
                self.unit = "tokens" if self.unit == "usd" else "usd"
                self.query_one("#unit_toggle", Static).update(self._render_unit())
                event.stop()
            elif event.key == "enter":
                try:
                    inputs = self._inputs()
                    result = compute_tuning(inputs, _pool(get_config()))
                    self.controller.push_screen(
                        TunePreviewScreen(self.controller, inputs, result)
                    )
                except (ValueError, TypeError) as exc:
                    self.query_one("#message", Static).update(f"[red]Error: {exc}[/red]")
                event.stop()
            elif event.key in ("escape", "b"):
                if self.controller:
                    self.controller.pop_screen()
                event.stop()

    class AdvancedTuneScreen(Screen):
        def __init__(self, controller=None):
            super().__init__()
            self.controller = controller
            self.cursor = 0
            self._editing = False
            self._message = ""
            cfg = get_config()
            self.current = {
                **_defaults(),
                "ambiguous_low": cfg.ambiguous_low,
                "ambiguous_high": cfg.ambiguous_high,
                "burst_factor": 1.8,
            }
            self.new = dict(self.current)

        def _range(self, key: str) -> str:
            if "bias" in key:
                return "[-0.5, 0.5]"
            if "gamma" in key:
                return "[0.1, 5.0]"
            if "alpha" in key or "quality" in key:
                return "(0, 1]"
            if "ambiguous" in key:
                return "[0, 1]"
            return ">= 0"

        def _table(self) -> str:
            keys = list(self.current)
            lines = [
                f"{'Param':<30} {'Default':<10} {'Range':<12} {'Current':<10} {'New':<10}",
                "-" * 74,
            ]
            for i, k in enumerate(keys):
                v_def = self.current[k]
                v_cur = self.current.get(k)
                v_new = self.new.get(k)
                row_str = f"{k:<30} {str(v_def):<10} {self._range(k):<12} {str(v_cur):<10} {str(v_new):<10}"
                mark = ">> " if i == self.cursor else "   "
                row = f"{mark}{row_str}"
                lines.append(f"[reverse]{row}[/reverse]" if i == self.cursor else row)
            lines.append("-" * 74)
            if self._message:
                lines.append(self._message)
            return "\n".join(lines)

        def compose(self):
            yield Vertical(
                Static("AutoConduck - Advanced Parameter Tuning"),
                Static(self._table(), id="table", markup=True),
                Input(placeholder="", id="param_edit"),
                Static(
                    "[up/down] select param  [left/right] adjust (±0.05)  [enter] edit/preview  [esc] back  [ctrl+c] quit",
                    id="help",
                    markup=False,
                ),
            )

        def on_mount(self):
            try:
                self.query_one("#param_edit", Input).display = False
            except Exception:
                pass

        def _adjust(self, delta: float):
            keys = list(self.current)
            key = keys[self.cursor]
            cur = float(self.new.get(key, 0.0))
            new_val = round(cur + delta, 2)
            if "ambiguous" in key:
                new_val = max(0.0, min(1.0, new_val))
            elif "bias" in key:
                new_val = max(-0.5, min(0.5, new_val))
            elif "gamma" in key:
                new_val = max(0.1, min(5.0, new_val))
            elif "alpha" in key or "quality" in key:
                new_val = max(0.01, min(1.0, new_val))
            self.new[key] = new_val
            self._message = f"Adjusted {key} to {new_val}"
            self.query_one("#table", Static).update(self._table())

        def on_key(self, event):
            keys = list(self.current)
            if self._editing:
                if event.key == "escape":
                    self._editing = False
                    try:
                        self.query_one("#param_edit", Input).display = False
                    except Exception:
                        pass
                    self._message = ""
                    self.query_one("#table", Static).update(self._table())
                return
            if event.key == "down":
                self.cursor = (self.cursor + 1) % len(keys)
                self.query_one("#table", Static).update(self._table())
            elif event.key == "up":
                self.cursor = (self.cursor - 1) % len(keys)
                self.query_one("#table", Static).update(self._table())
            elif event.key in ("right", "+", "="):
                self._adjust(0.05)
            elif event.key in ("-",):
                self._adjust(-0.05)
            elif event.key == "left":
                self._adjust(-0.05)
            elif event.key in ("enter", "p"):
                try:
                    inputs = SimpleInputs(87)
                    result = compute_tuning(
                        inputs, _pool(get_config()), current=self.new
                    )
                    self.controller.push_screen(
                        TunePreviewScreen(self.controller, inputs, result)
                    )
                except Exception as exc:
                    self._message = f"[red]Error: {exc}[/red]"
                    self.query_one("#table", Static).update(self._table())
            elif event.key in ("escape", "b"):
                if self.controller:
                    self.controller.pop_screen()

    class TunePreviewScreen(Screen):
        def __init__(self, controller, inputs, result):
            super().__init__()
            self.controller = controller
            self.inputs = inputs
            self.result = result

        def compose(self):
            lines = [
                "AutoConduck - Proposed Tuning Profile",
                "",
                f"{'Tunable':<30} | {'Current':<15} | {'Proposed':<15}",
                "-" * 65,
            ] + [
                f"{k:<30} | {str(a):<15} | {str(b):<15}"
                for k, (a, b) in self.result.tunables.items()
            ]
            lines += [
                "",
                "Warnings: " + ("; ".join(self.result.warnings) or "none"),
                "c: commit changes · r: reset to defaults · esc/left: back · [ctrl+c] quit",
            ]
            yield Static("\n".join(lines))

        def on_key(self, event):
            if event.key == "c":
                _commit(self.result, self.inputs)
                self.controller.pop_screen()
            elif event.key == "r":
                self.result = compute_tuning(
                    self.inputs, _pool(get_config()), current=_defaults()
                )
                self.refresh()
            elif event.key in ("escape", "left", "b"):
                self.controller.pop_screen()

else:

    class TuneScreen:
        def __init__(self, *args, **kwargs):
            _require_textual()

    SimpleTuneScreen = AdvancedTuneScreen = TunePreviewScreen = TuneScreen
