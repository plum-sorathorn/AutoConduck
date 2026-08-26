"""Headless TUI import and rendering smoke test."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    try:
        from autoconduck.tui.dashboard import DashboardScreen, MainMenuScreen
        from autoconduck.tui.dashboard_widgets import _format_box_lines, render_log_rows
        from autoconduck.tui.keymap import KEYMAP, QUIT_KEY
        from autoconduck.tui.onboarding.screens_models import ModelCatalogScreen
        from autoconduck.tui.settings import SettingsScreen
        for name, value in (("MainMenuScreen", MainMenuScreen), ("DashboardScreen", DashboardScreen), ("ModelCatalogScreen", ModelCatalogScreen), ("SettingsScreen", SettingsScreen)):
            value()
            print(f"PASS: {name} constructed")
        assert QUIT_KEY in KEYMAP and "down" in KEYMAP
        assert isinstance(render_log_rows([], 0), str) and isinstance(_format_box_lines("x", [], 12), list)
        print("PASS: keybindings and rendering helpers")
        return 0
    except Exception as exc:
        print(f"FAIL: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
