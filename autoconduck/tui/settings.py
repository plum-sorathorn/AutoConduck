"""Settings screen — expose configurable options from the TUI."""

from __future__ import annotations
from typing import Any

from .onboarding import _TEXTUAL, _require_textual
from autoconduck.config import get_config, save_config


if _TEXTUAL:
    from textual.screen import Screen
    from textual.containers import Vertical
    from textual.widgets import Static, Input

    class SettingsScreen(Screen):
        """TUI settings page."""

        SETTINGS_DEFS = [
            (
                "launch_in_new_terminal",
                "Launch proxy in new terminal",
                "conduck start --agent opens the proxy in a new window",
                "bool",
            ),
            (
                "port",
                "Proxy listen port",
                "Port the AutoConduck proxy binds to (default 11434)",
                "int",
            ),
            (
                "log_level",
                "Log level",
                "Proxy verbosity: DEBUG / INFO / WARNING / ERROR",
                "str",
            ),
            (
                "selection.slm_circuit_breaker_timeout_ms",
                "SLM Circuit Breaker (ms)",
                "Timeout before degrading to balanced fallback (default 100)",
                "int",
            ),
            (
                "selection.slm_model_path",
                "Embedded SLM Model Path",
                "Path to local ONNX/embedded SLM model",
                "str",
            ),
            (
                "selection.session_guard_compaction_ratio",
                "Session Compaction Ratio",
                "Context ceiling ratio before session compaction (default 0.80)",
                "float",
            ),
            (
                "selection.rag_max_tokens",
                "RAG Retrieval Max Tokens",
                "Max retrieved tokens from LanceDB vector store (default 250)",
                "int",
            ),
        ]

        LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"]

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

        def _set_val(self, path: str, value: Any):
            cfg = get_config()
            parts = path.split(".")
            parent = cfg
            for p in parts[:-1]:
                parent = getattr(parent, p)
            key = parts[-1]
            setattr(parent, key, value)
            save_config(cfg)

        def _toggle_bool(self, path: str, name: str):
            cur = bool(self._get_val(path))
            new_val = not cur
            self._set_val(path, new_val)
            self._message = f"{name}: {new_val}"
            self.query_one("#rows").update(self._render_rows())

        def _cycle_log_level(self):
            cur = str(self._get_val("log_level") or "INFO").upper()
            idx = self.LOG_LEVELS.index(cur) if cur in self.LOG_LEVELS else 1
            new_val = self.LOG_LEVELS[(idx + 1) % len(self.LOG_LEVELS)]
            self._set_val("log_level", new_val)
            self._message = f"Log level: {new_val}"
            self.query_one("#rows").update(self._render_rows())

        def _adjust_float(self, path: str, name: str, delta: float):
            cur = float(self._get_val(path) or 0.0)
            new_val = round(max(0.0, min(1.0, cur + delta)), 2)
            self._set_val(path, new_val)
            self._message = f"{name}: {new_val:.2f}"
            self.query_one("#rows").update(self._render_rows())

        def _render_rows(self):
            lines = []
            for i, (path, name, desc, type_hint) in enumerate(self.SETTINGS_DEFS):
                val = self._get_val(path)
                mark = ">> " if i == self.cursor else "   "
                if type_hint == "bool":
                    val_str = "[green]True[/green]" if val else "[dim]False[/dim]"
                elif type_hint == "float" and isinstance(val, (int, float)):
                    val_str = f"{float(val):.2f}"
                else:
                    val_str = str(val)
                row = f"{mark}{name}  \\[{val_str}]"
                lines.append(f"[reverse]{row}[/reverse]" if i == self.cursor else row)
            if self._message:
                lines.append(f"\n{self._message}")
            return "\n".join(lines)

        def compose(self):
            yield Vertical(
                Static("AutoConduck - Settings"),
                Static(self._render_rows(), id="rows", markup=True),
                Input(placeholder="", id="edit_input"),
                Static(
                    "[up/down] navigate  [enter/space] toggle/cycle  [left/right] adjust (±0.05)  [esc] back  [ctrl+c] quit",
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

        def _commit_edit(self, raw: str):
            path, name, desc, type_hint = self.SETTINGS_DEFS[self.cursor]
            try:
                if type_hint == "int":
                    val = int(raw.strip())
                elif type_hint == "float":
                    raw_val = float(raw.strip())
                    # Snap to nearest 0.05 increment and cap at 1.0
                    val = round(round(max(0.0, min(1.0, raw_val)) / 0.05) * 0.05, 2)
                else:
                    val = raw.strip()
                self._set_val(path, val)
                self._message = f"Saved {name}: {val}"
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
            path, name, desc, type_hint = self.SETTINGS_DEFS[self.cursor]
            if event.key == "down":
                self.cursor = min(self.cursor + 1, len(self.SETTINGS_DEFS) - 1)
                self.query_one("#rows").update(self._render_rows())
            elif event.key == "up":
                self.cursor = max(self.cursor - 1, 0)
                self.query_one("#rows").update(self._render_rows())
            elif event.key in ("enter", "space"):
                if type_hint == "bool":
                    self._toggle_bool(path, name)
                elif path == "log_level":
                    self._cycle_log_level()
                elif type_hint == "float":
                    self._adjust_float(path, name, 0.05)
                elif type_hint == "int":
                    self._start_edit()
            elif event.key in ("right", "+", "="):
                if type_hint == "float":
                    self._adjust_float(path, name, 0.05)
            elif event.key in ("left", "-"):
                if type_hint == "float":
                    self._adjust_float(path, name, -0.05)
                else:
                    if self.controller:
                        self.controller.pop_screen()
            elif event.key in ("escape", "b"):
                if self.controller:
                    self.controller.pop_screen()
            elif event.key == "e":
                self._start_edit()
            elif event.key == "c" and self.controller:
                from .onboarding import ModelSourceScreen

                self.controller.push_screen(ModelSourceScreen(self.controller))
            event.stop()

else:

    class SettingsScreen:
        def __init__(self, *args, **kwargs):
            _require_textual()
