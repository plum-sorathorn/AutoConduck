"""Shared keyboard conventions for every TUI screen."""
KEYMAP = {
    "down": ("move_down", "move selection down"),
    "up": ("move_up", "move selection up"),
    "right": ("forward", "continue/advance"), "left": ("back", "go back"),
    "enter": ("toggle", "toggle selection"), "space": ("toggle", "toggle selection"), "ctrl+s": ("save", "save"),
    "/": ("filter", "focus search"),
    "?": ("help", "keybind reference"),
    "ctrl+c": ("quit", "quit current screen / quit app from top-level"),
    "d": ("drill", "drill selected decision"), "p": ("pause", "pause/resume routing"), "e": ("edit", "edit models"),
    "m": ("models", "open model configuration"), "u": ("update", "check for updates"), "s": ("settings", "open settings"), "a": ("agent", "launch agent"),
}

QUIT_KEY = "ctrl+c"

def FOOTER_HINT(*keys: str) -> str:
    """Render a compact footer; with no arguments render the global set."""
    selected = keys or ("↑/↓", "enter", "d", "p", "left", QUIT_KEY)
    return "  ".join(
        f"[{key}] {'quit' if key == QUIT_KEY else KEYMAP.get(key, (key, key))[1]}"
        for key in selected
    )
