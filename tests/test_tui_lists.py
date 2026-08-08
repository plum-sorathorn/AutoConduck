from autoconduck.tui.onboarding import render_agent_rows, render_source_rows, move_cursor
from autoconduck.tui.dashboard import render_log_rows

def test_agent_rows_cursor_and_selection():
    text = render_agent_rows(["a", "b"], {"a": "/a", "b": None}, {"b"}, 1)
    assert text.count("›") == 1 and "✓" in text
    assert text.index("›") > text.index("a")

def test_source_rows_cursor_selected_and_pool():
    text = render_source_rows(["A", "B"], {1}, 0, ["gpt-4"])
    assert text.count("›") == 1 and "✓" in text and "gpt-4" in text

def test_log_rows():
    assert "no routing decisions" in render_log_rows([], 0)
    text = render_log_rows([{"time": "12:00", "route": "slow", "model": "m", "prompt": "hello", "confidence": .8}], 0)
    assert text.count("›") == 1 and all(value in text for value in ("12:00", "slow", "m", "hello", ".8"))

def test_cursor_clamps():
    assert move_cursor(0, -1, 3) == 0
    assert move_cursor(2, 1, 3) == 2
