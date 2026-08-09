import argparse
import json
from types import SimpleNamespace

from autoconduck import stats
from autoconduck import main


def test_stats_path_uses_home_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOCONDUCK_HOME", str(tmp_path))
    assert stats.stats_path() == tmp_path / "run" / "stats.jsonl"


def test_stats_record_and_aggregate(monkeypatch, tmp_path):
    path = tmp_path / "stats.jsonl"
    monkeypatch.setattr(stats, "stats_path", lambda: path)
    monkeypatch.setattr(stats, "estimate_cost", lambda m, p, c: float(p + c) / 100)
    stats.record("FAST", "autoconduck", "one", 10, 5)
    stats.record("SLOW", "autoconduck", "two", 20, 3)
    agg = stats.aggregate(stats.load_records())
    assert agg["totals"]["calls"] == 2
    assert agg["totals"]["total_tokens"] == 38
    assert agg["models"]["one"]["cost"] == 0.15
    assert agg["paths"] == {"FAST": 1, "SLOW": 1}


def test_stats_cost_math(monkeypatch):
    monkeypatch.setattr(stats.pricing, "_entry", lambda model: {"price_in": 2.5, "price_out": 10.0})
    assert stats.estimate_cost("x", 1_000_000, 500_000) == 7.5


def test_cmd_stats_cli(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(stats, "stats_path", lambda: tmp_path / "stats.jsonl")
    stats.record("FAST", "autoconduck", "model-x", 1, 2)
    main.cmd_stats(argparse.Namespace(json=False, days=None, reset=False, force=False))
    assert "model-x" in capsys.readouterr().out
    main.cmd_stats(argparse.Namespace(json=True, days=None, reset=False, force=False))
    assert json.loads(capsys.readouterr().out)["totals"]["calls"] == 1


def test_stats_reset_requires_force(monkeypatch, tmp_path):
    path = tmp_path / "stats.jsonl"
    path.write_text("{}\n")
    monkeypatch.setattr(stats, "stats_path", lambda: path)
    main.cmd_stats(argparse.Namespace(json=False, days=None, reset=True, force=False))
    assert path.exists()
    main.cmd_stats(argparse.Namespace(json=False, days=None, reset=True, force=True))
    assert not path.exists()
