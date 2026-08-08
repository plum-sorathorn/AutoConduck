"""Shared keyboard conventions for every TUI screen."""
KEYMAP = {
    "j": ("move_down", "move selection down"), "down": ("move_down", "move selection down"),
    "k": ("move_up", "move selection up"), "up": ("move_up", "move selection up"),
    "enter": ("select", "select/toggle"), "space": ("select", "select/toggle"),
    "esc": ("back", "back/close/cancel"), "/": ("filter", "inline filter"),
    "?": ("help", "keybind reference"),
    "ctrl+c": ("quit", "quit current screen / quit app from top-level"),
    "d": ("drill", "drill"), "p": ("pause", "pause/resume routing"), "e": ("edit", "edit models"),
}

QUIT_KEY = "ctrl+c"

def FOOTER_HINT(*keys: str) -> str:
    """Render a compact footer; with no arguments render the global set."""
    selected = keys or ("j/k", "enter/space", "esc", "/", "?", QUIT_KEY, "p", "e")
    return "  ".join(
        f"[{key}] {'quit' if key == QUIT_KEY else KEYMAP.get(key, (key, key))[1]}"
        for key in selected
    )
