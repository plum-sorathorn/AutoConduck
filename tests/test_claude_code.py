import json

from autoconduck.agents.claude_code import ClaudeCodeAdapter
from autoconduck.config import Config


def test_patch_and_revert_preserve_user_settings_and_backup(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: tmp_path))
    monkeypatch.setenv("AUTOCONDUCK_HOME", str(tmp_path / ".autoconduck"))
    settings = tmp_path / ".claude" / "settings.json"
    settings.parent.mkdir()
    settings.write_text(json.dumps({"theme": "dark", "env": {"USER_KEY": "keep"}}))
    adapter = ClaudeCodeAdapter()

    adapter.patch(Config(pseudo_model="autoconduck-budget"), 1234)
    first = json.loads(settings.read_text())
    assert first["theme"] == "dark"
    assert first["env"]["USER_KEY"] == "keep"
    assert first["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:1234"
    assert first["env"]["ANTHROPIC_AUTH_TOKEN"] == "autoconduck-local"
    assert first["env"]["ANTHROPIC_MODEL"] == "autoconduck-budget"
    assert first["autoconduck"]
    assert list((tmp_path / ".autoconduck" / "backups" / "claude_code").glob("*.bak"))

    adapter.patch(Config(), 1234)
    assert json.loads(settings.read_text())["env"]["USER_KEY"] == "keep"
    adapter.revert()
    final = json.loads(settings.read_text())
    assert final == {"theme": "dark", "env": {"USER_KEY": "keep"}}
