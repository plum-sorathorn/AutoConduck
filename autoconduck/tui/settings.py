"""Settings screen — expose configurable options from the TUI."""
from __future__ import annotations

from .onboarding import _TEXTUAL, _require_textual
from autoconduck.config import get_config, save_config


if _TEXTUAL:
    from textual.screen import Screen
    from textual.containers import Vertical
    from textual.widgets import Static, Input

    class SettingsScreen(Screen):
        """TUI settings page."""

        SETTINGS_DEFS = [
            ("launch_in_new_terminal", "Launch proxy in new terminal",
             "conduck start --agent opens the proxy in a new window", "bool"),
            ("ambiguous_low",          "Ambiguous band lower edge",
             "Confidence below this => ambiguous zone (default 0.55)", "float"),
            ("ambiguous_high",         "Ambiguous band upper edge",
             "Confidence above this => confident routing (default 0.70)", "float"),
            ("selection.slow_threshold", "Slow-path threshold",
             "Complexity at or above this => SLOW path (default 0.75)", "float"),
            ("port",                   "Proxy listen port",
             "Port the AutoConduck proxy binds to (default 11434)", "int"),
            ("log_level",              "Log level",
             "Proxy verbosity: DEBUG / INFO / WARNING / ERROR", "str"),
            ("selection.tiebreaker_enabled", "Tiebreaker enabled",
             "Use a cheap LLM to break routing ties in the ambiguous band", "bool"),
            ("selection.tiebreaker_min_complexity", "Tiebreaker min complexity",
             "Only invoke the tiebreaker when complexity >= this value (default 0.45)", "float"),
        ]

        def __init__(self, controller=None):
            super().__init__()
            self.controller = controller
            self.cursor = 0
            self._editing = False
            self._message = ""

        def _get_val(self, path: str):
            cfg = get_config()
            parts = path.split(".")
            obj = cfg
            for p in parts:
                obj = getattr(obj, p, None)
                if obj is None:
                    return None
            return obj

        def _set_val(self, path: str, raw: str, type_hint: str):
            cfg = get_config()
            parts = path.split(".")
            parent = cfg
            for p in parts[:-1]:
                parent = getattr(parent, p)
            key = parts[-1]
            if type_hint == "bool":
                value = raw.strip().lower() in ("true", "1", "yes", "on")
            elif type_hint == "int":
                value = int(raw.strip())
            elif type_hint == "float":
                value = float(raw.strip())
            else:
                value = raw.strip()
            setattr(parent, key, value)
            save_config(cfg)

        def _render_rows(self):
            lines = []
            for i, (path, name, desc, _) in enumerate(self.SETTINGS_DEFS):
                val = self._get_val(path)
                mark = ">> " if i == self.cursor else "   "
                row = f"{mark}{name}  \\[{val}]"
                lines.append(
                    f"[reverse]{row}[/reverse]" if i == self.cursor else row
                )
            if self._message:
                lines.append(f"\n{self._message}")
            return "\n".join(lines)

        def compose(self):
            yield Vertical(
                Static("AutoConduck - Settings"),
                Static(self._render_rows(), id="rows", markup=True),
                Input(placeholder="", id="edit_input"),
                Static(
                    "[up/down] navigate  [e/enter] edit  [left/esc] back  [ctrl+c] quit",
                    id="footer",
                    markup=False,
                ),
            )

        def on_mount(self):
            try:
                self.query_one("#edit_input", Input).display = False
            except Exception:
                pass

        def _start_edit(self):
            path, name, desc, type_hint = self.SETTINGS_DEFS[self.cursor]
            val = self._get_val(path)
            self._editing = True
            self._message = f"Editing: {name} ({desc})"
            try:
                inp = self.query_one("#edit_input", Input)
                inp.value = str(val) if val is not None else ""
                inp.placeholder = f"new value for {name}"
                inp.display = True
                inp.focus()
            except Exception:
                pass
            self.query_one("#rows").update(self._render_rows())

        def _commit_edit(self, value: str):
            path, name, desc, type_hint = self.SETTINGS_DEFS[self.cursor]
            try:
                self._set_val(path, value, type_hint)
                self._message = f"Saved {name}"
            except (ValueError, TypeError) as exc:
                self._message = f"Error: {exc}"
            finally:
                self._editing = False
                try:
                    inp = self.query_one("#edit_input", Input)
                    inp.display = False
                except Exception:
                    pass
                self.query_one("#rows").update(self._render_rows())

        def on_input_submitted(self, event):
            if event.input.id == "edit_input" and self._editing:
                self._commit_edit(event.value)
                event.stop()

        def on_key(self, event):
            if self._editing:
                if event.key in ("escape",):
                    self._editing = False
                    self._message = ""
                    try:
                        self.query_one("#edit_input", Input).display = False
                    except Exception:
                        pass
                    self.query_one("#rows").update(self._render_rows())
                return
            if event.key == "down":
                self.cursor = min(self.cursor + 1, len(self.SETTINGS_DEFS) - 1)
                self.query_one("#rows").update(self._render_rows())
            elif event.key == "up":
                self.cursor = max(self.cursor - 1, 0)
                self.query_one("#rows").update(self._render_rows())
            elif event.key in ("enter", "e"):
                self._start_edit()
            elif event.key in ("left", "escape"):
                if self.controller:
                    self.controller.pop_screen()
            event.stop()

else:
    class SettingsScreen:
        def __init__(self, *args, **kwargs):
            _require_textual()
