import json

from autoconduck.agents.claude_code import ClaudeCodeAdapter
from autoconduck.config import ClaudeCodeSettings, Config


def _setup(monkeypatch, tmp_path, content=None):
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    path = tmp_path / ".claude" / "settings.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(content or {}), encoding="utf-8")
    return path, ClaudeCodeAdapter()


def test_patch_merges_default_and_existing_permissions(monkeypatch, tmp_path):
    path, adapter = _setup(monkeypatch, tmp_path, {"permissions": {"allow": ["Notebook"]}})
    adapter.patch(Config(), port=1234)
    permissions = json.loads(path.read_text(encoding="utf-8"))["permissions"]
    assert permissions["allow"] == ["Notebook", "Task", "Skill", "Git", "Read", "Write", "Edit", "WebFetch"]
    assert "Bash" not in permissions["allow"]


def test_revert_restores_previous_permissions_exactly(monkeypatch, tmp_path):
    original = {"allow": ["Notebook"], "deny": ["Bash"], "defaultMode": "plan"}
    path, adapter = _setup(monkeypatch, tmp_path, {"permissions": original})
    adapter.patch(Config())
    adapter.revert()
    assert json.loads(path.read_text(encoding="utf-8"))["permissions"] == original


def test_revert_removes_contributed_permissions_without_prior_block(monkeypatch, tmp_path):
    path, adapter = _setup(monkeypatch, tmp_path, {"theme": "dark"})
    adapter.patch(Config())
    adapter.revert()
    assert json.loads(path.read_text(encoding="utf-8")) == {"theme": "dark"}


def test_custom_allowed_tools_are_honored(monkeypatch, tmp_path):
    path, adapter = _setup(monkeypatch, tmp_path)
    cfg = Config(claude_code=ClaudeCodeSettings(allowed_tools=["CustomTool"]))
    adapter.patch(cfg)
    assert json.loads(path.read_text(encoding="utf-8"))["permissions"]["allow"] == ["CustomTool"]


def test_patch_preserves_existing_env_merge(monkeypatch, tmp_path):
    path, adapter = _setup(monkeypatch, tmp_path, {"env": {"KEEP": "value"}})
    adapter.patch(Config())
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["env"]["KEEP"] == "value"
    assert "ANTHROPIC_BASE_URL" in data["env"]
