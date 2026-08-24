"""Coding agent adapters unit tests for Pi, Claude Code, OpenCode, and all supported tools."""

import json
from pathlib import Path

from autoconduck.harnesses import all_adapters, all_harnesses, binary_name_for
from autoconduck.harnesses.aider import AiderAdapter
from autoconduck.harnesses.claude_code import ClaudeCodeAdapter
from autoconduck.harnesses.continue_dev import ContinueDevAdapter
from autoconduck.harnesses.cursor import CursorAdapter
from autoconduck.harnesses.generic_openai import GenericOpenAIAdapter
from autoconduck.harnesses.kilocode import KiloCodeAdapter
from autoconduck.harnesses.opencode import OpenCodeAdapter
from autoconduck.harnesses.pi import PiAdapter
from autoconduck.cli.cli_launch import resolve_agent_ids
from autoconduck.config import Config
from autoconduck.launcher import shim_script, shim_script_win
from autoconduck.presets.model_presets import curated_model_catalog, discover_models
from autoconduck.presets.presets_data import PRESETS


def test_all_adapters_registered():
    adapters = all_adapters()
    ids = {a.id for a in adapters}
    expected = {
        "claude_code",
        "opencode",
        "pi",
        "aider",
        "cursor",
        "continue_dev",
        "kilocode",
        "generic_openai",
    }
    assert expected.issubset(ids)
    assert binary_name_for("claude_code") == "claude"
    assert binary_name_for("opencode") == "opencode"
    assert binary_name_for("pi") == "pi"


def test_harnesses_module_alias():
    import autoconduck.harnesses as harnesses
    adapters = harnesses.all_harnesses()
    ids = {a.id for a in adapters}
    assert "claude_code" in ids
    assert "opencode" in ids
    assert "pi" in ids
    assert "aider" in ids
    assert "cursor" in ids
    assert "continue_dev" in ids
    assert "kilocode" in ids
    assert "generic_openai" in ids


def test_claude_code_adapter_patch_and_revert(tmp_path, monkeypatch):
    settings_file = tmp_path / ".claude" / "settings.json"
    settings_file.parent.mkdir(parents=True, exist_ok=True)
    settings_file.write_text(
        json.dumps(
            {
                "env": {"EXISTING": "val"},
                "permissions": {"allow": ["Notebook", "Task", "Read", "Write", "Edit"]},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    adapter = ClaudeCodeAdapter()
    cfg = Config(port=11434)
    adapter.patch(cfg, port=11434)

    # Check that settings were patched
    data = json.loads(settings_file.read_text(encoding="utf-8"))
    assert "ANTHROPIC_BASE_URL" in data["env"]
    assert "http://127.0.0.1:11434" in data["env"]["ANTHROPIC_BASE_URL"]
    overrides = data.get("modelOverrides", {})
    assert "autoconduck" in overrides
    # modelOverrides values must be strings (provider model IDs), not objects.
    for pseudo_name in ("autoconduck", "autoconduck-budget", "autoconduck-expensive"):
        assert overrides[pseudo_name] == pseudo_name
        assert isinstance(overrides[pseudo_name], str)

    # Revert restores original state
    adapter.revert()
    data_reverted = json.loads(settings_file.read_text(encoding="utf-8"))
    assert "ANTHROPIC_BASE_URL" not in data_reverted.get("env", {})
    assert data_reverted.get("env", {}).get("EXISTING") == "val"


def test_pi_adapter_patch_install_features_and_revert(tmp_path, monkeypatch):
    pi_dir = tmp_path / ".pi" / "agent"
    pi_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(pi_dir))

    adapter = PiAdapter()
    cfg = Config(port=11434)
    adapter.patch(cfg, port=11434)

    ext = pi_dir / "extensions" / "autoconduck.ts"
    assert ext.exists()
    assert "autoconduck" in ext.read_text(encoding="utf-8")

    # Check provider configuration in settings.json
    settings_file = pi_dir / "settings.json"
    assert settings_file.exists()
    settings_data = json.loads(settings_file.read_text(encoding="utf-8"))
    assert settings_data.get("defaultProvider") == "autoconduck"

    # Test install_features hook
    installed = adapter.install_features()
    assert isinstance(installed, list)

    adapter.revert()
    assert not ext.exists()
    data_reverted = json.loads(settings_file.read_text(encoding="utf-8"))
    assert data_reverted.get("defaultProvider") is None


def test_opencode_adapter_patch_and_revert(tmp_path, monkeypatch):
    opencode_cfg = tmp_path / "opencode.json"
    opencode_cfg.write_text(json.dumps({"providers": {}}), encoding="utf-8")
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    adapter = OpenCodeAdapter()
    cfg = Config(port=11434)
    adapter.patch(cfg, port=11434)

    data = json.loads(opencode_cfg.read_text(encoding="utf-8"))
    assert "autoconduck" in data.get("providers", {})
    assert "autoconduck" in data.get("provider", {})
    assert data.get("model") == "autoconduck/autoconduck"

    adapter.revert()
    data_reverted = json.loads(opencode_cfg.read_text(encoding="utf-8"))
    assert "autoconduck" not in data_reverted.get("providers", {})
    assert "autoconduck" not in data_reverted.get("provider", {})


def test_aider_adapter_patch_and_revert(tmp_path, monkeypatch):
    aider_cfg = tmp_path / ".aider.conf.yml"
    aider_cfg.write_text("verbose: true\n", encoding="utf-8")
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    adapter = AiderAdapter()
    cfg = Config(port=11434)
    adapter.patch(cfg, port=11434)

    text = aider_cfg.read_text(encoding="utf-8")
    assert "openai_api_base: http://127.0.0.1:11434/v1" in text
    assert "model: openai/autoconduck" in text

    adapter.revert()
    text_reverted = aider_cfg.read_text(encoding="utf-8")
    assert "openai_api_base" not in text_reverted
    assert "verbose: true" in text_reverted


def test_cursor_adapter_patch_and_revert(tmp_path, monkeypatch):
    cursor_dir = tmp_path / ".cursor"
    cursor_dir.mkdir(parents=True, exist_ok=True)
    cursor_cfg = cursor_dir / "settings.json"
    cursor_cfg.write_text(json.dumps({"editor.fontSize": 14}), encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    adapter = CursorAdapter()
    cfg = Config(port=11434)
    adapter.patch(cfg, port=11434)

    data = json.loads(cursor_cfg.read_text(encoding="utf-8"))
    assert "autoconduck" in data
    assert data.get("openai.apiBase") == "http://127.0.0.1:11434/v1"

    adapter.revert()
    data_reverted = json.loads(cursor_cfg.read_text(encoding="utf-8"))
    assert "autoconduck" not in data_reverted
    assert data_reverted.get("editor.fontSize") == 14


def test_continue_adapter_patch_and_revert(tmp_path, monkeypatch):
    continue_dir = tmp_path / ".continue"
    continue_dir.mkdir(parents=True, exist_ok=True)
    continue_cfg = continue_dir / "config.json"
    continue_cfg.write_text(
        json.dumps({"models": [{"title": "existing", "model": "gpt-4"}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    adapter = ContinueDevAdapter()
    cfg = Config(port=11434)
    adapter.patch(cfg, port=11434)

    data = json.loads(continue_cfg.read_text(encoding="utf-8"))
    titles = [m.get("title") for m in data.get("models", [])]
    assert "autoconduck" in titles
    assert "autoconduck-budget" in titles

    adapter.revert()
    data_reverted = json.loads(continue_cfg.read_text(encoding="utf-8"))
    reverted_titles = [m.get("title") for m in data_reverted.get("models", [])]
    assert "autoconduck" not in reverted_titles
    assert "existing" in reverted_titles


def test_kilocode_adapter_patch_and_revert(tmp_path, monkeypatch):
    kilo_cfg = tmp_path / "kilo-config.json"
    kilo_cfg.write_text(json.dumps({"user": "dev"}), encoding="utf-8")
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    adapter = KiloCodeAdapter()
    cfg = Config(port=11434)
    adapter.patch(cfg, port=11434)

    data = json.loads(kilo_cfg.read_text(encoding="utf-8"))
    assert data.get("api_base") == "http://127.0.0.1:11434/v1"

    adapter.revert()
    data_reverted = json.loads(kilo_cfg.read_text(encoding="utf-8"))
    assert "autoconduck" not in data_reverted
    assert data_reverted.get("user") == "dev"


def test_generic_openai_adapter():
    adapter = GenericOpenAIAdapter()
    assert adapter.detect() is True
    cfg = Config(port=11434)
    adapter.patch(cfg)
    adapter.revert()


def test_launcher_shims_generation():
    real_bin = "/usr/local/bin/agent_mock"
    for agent_id in (
        "claude_code",
        "opencode",
        "pi",
        "aider",
        "cursor",
        "continue_dev",
        "kilocode",
    ):
        bash = shim_script(agent_id, real_bin)
        assert "ensure --port" in bash
        assert "release --port" in bash

        win = shim_script_win(agent_id, real_bin)
        assert "ensure --port" in win
        assert "release --port" in win


def test_agent_alias_resolution():
    assert resolve_agent_ids(["claude"]) == ["claude_code"]
    assert resolve_agent_ids(["open-code"]) == ["opencode"]
    assert resolve_agent_ids(["continue"]) == ["continue_dev"]
    assert resolve_agent_ids(["kilo-code"]) == ["kilocode"]
    assert resolve_agent_ids(["all"]) == [a.id for a in all_adapters()]


def test_all_provider_presets_resolve():
    for provider in (
        "openai",
        "anthropic",
        "google",
        "mistral",
        "deepseek",
        "groq",
        "openrouter",
        "together",
        "xai",
        "llmgateway",
        "devpass",
    ):
        assert provider in PRESETS
        models = PRESETS[provider]
        assert len(models) >= 1
        for m in models:
            assert "id" in m
            assert "provider" in m
            assert "tier" in m
            assert "price_in" in m
            assert "price_out" in m

    discovered = discover_models(
        preset_keys=[
            "openai",
            "anthropic",
            "google",
            "mistral",
            "deepseek",
            "groq",
            "openrouter",
            "together",
            "xai",
        ]
    )
    assert len(discovered) >= 40
    catalog = curated_model_catalog()
    assert len(catalog) >= 100

    # Verify grok-4.6 presence in xai and devpass
    assert any(m["id"] == "grok-4.6" for m in PRESETS["xai"])
    assert any(m["id"] == "grok-4.6" for m in PRESETS["devpass"])


def test_claude_code_sanitizes_legacy_object_model_overrides(tmp_path, monkeypatch):
    """Ensure legacy object modelOverrides values are sanitized into strings."""
    settings_file = tmp_path / ".claude" / "settings.json"
    settings_file.parent.mkdir(parents=True, exist_ok=True)
    settings_file.write_text(
        json.dumps(
            {
                "env": {"EXISTING": "val"},
                "modelOverrides": {
                    "autoconduck": {"contextWindow": 1000000},
                    "autoconduck-budget": {"contextWindow": 1000000},
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    adapter = ClaudeCodeAdapter()
    cfg = Config(port=11434)
    adapter.patch(cfg, port=11434)

    data = json.loads(settings_file.read_text(encoding="utf-8"))
    overrides = data.get("modelOverrides", {})
    for pseudo_name in ("autoconduck", "autoconduck-budget", "autoconduck-expensive"):
        assert overrides[pseudo_name] == pseudo_name
        assert isinstance(overrides[pseudo_name], str)


def test_onboarding_configure_selected_agents(tmp_path, monkeypatch):
    from autoconduck.tui.onboarding.helpers import configure_selected_agents

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / ".pi" / "agent"))

    configured = configure_selected_agents(["claude_code", "pi"])
    assert "claude_code" in configured
    assert "pi" in configured


def test_universal_handoff_execution_directives(tmp_path, monkeypatch):
    from autoconduck.orchestrator.handoff import format_execution_handoff
    from autoconduck.routing.slm_planner import ExecutionPlan, SubTaskSpec

    plan = ExecutionPlan(
        summary="Test multi-step plan",
        subtasks=[
            SubTaskSpec(id="recon", goal="Scan repo", role="recon", scope=[], constraints=[]),
            SubTaskSpec(id="edit", goal="Apply patch", role="edit", scope=[], constraints=[], depends_on=["recon"]),
        ],
    )
    handoff = format_execution_handoff(
        plan=plan,
        subagent_outputs={"recon": "Found files"},
        compacted="",
        client_type="pi",
    )
    assert handoff.tool_calls is None
    assert "## Implementation Plan & Verified Context" in handoff
    assert "Found files" in handoff
    assert "Proceed with implementation of the subtasks sequentially" in handoff

