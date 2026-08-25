"""Model catalog and vault browser."""
from __future__ import annotations

from ..onboarding_models import catalog_filter_chips, filter_catalog
from autoconduck.config import get_config
from autoconduck.presets.model_presets import PRESETS

try:
    from textual.containers import Vertical
    from textual.screen import Screen
    from textual.widgets import Input, Static
    _TEXTUAL = True
except ImportError:
    _TEXTUAL = False
    class Screen: pass


if _TEXTUAL:
    class ModelCatalogScreen(Screen):
        TABS = ("Providers/Presets", "Custom Endpoints", "API Key Vault")

        def __init__(self, controller=None):
            super().__init__()
            self.controller = controller
            self.tab = 0
            self.models = [dict(row) for rows in PRESETS.values() for row in rows]
            self.filtered = self.models

        def compose(self):
            yield Vertical(Static(self._tabs(), id="tabs", markup=True),
                           Input(placeholder="search providers or models…", id="search"),
                           Static(self._body(), id="body", markup=True),
                           Static("[←] back  [tab/→] next tab  [/] search  [ctrl+c] quit", id="footer"))

        def _tabs(self):
            return "  ".join(f"[bold cyan]{tab}[/bold cyan]" if i == self.tab else tab for i, tab in enumerate(self.TABS))

        def _body(self):
            if self.tab == 0:
                chips = catalog_filter_chips(self.models)
                return "Providers: " + ", ".join(chips["providers"]) + "\n" + "\n".join(
                    f"{m.get('provider', '')}/{m.get('id', '')}" for m in self.filtered[:80]) or "No models available."
            if self.tab == 1:
                rows = getattr(get_config(), "custom_models", [])
                return "\n".join(f"{m.get('provider', '')}/{m.get('id', '')}  {m.get('base_url', '')}" for m in rows) or "No custom endpoints."
            rows = getattr(get_config(), "model_list", []) + getattr(get_config(), "custom_models", [])
            return "\n".join(f"{m.get('provider', '')}: {'configured' if m.get('api_key') or m.get('api_key_env') else 'missing'}" for m in rows) or "No API keys configured."

        def on_input_changed(self, event):
            if event.input.id == "search":
                self.filtered = filter_catalog(self.models, event.value)
                self.query_one("#body", Static).update(self._body())

        def on_key(self, event):
            if event.key in ("tab", "right"):
                self.tab = (self.tab + 1) % len(self.TABS)
            elif event.key == "left":
                self.tab = (self.tab - 1) % len(self.TABS)
            elif event.key == "escape":
                self.app.pop_screen()
            else:
                return
            self.query_one("#tabs", Static).update(self._tabs())
            self.query_one("#body", Static).update(self._body())
else:
    ModelCatalogScreen = Screen
