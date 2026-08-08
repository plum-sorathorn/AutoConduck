import tempfile
from pathlib import Path
from unittest.mock import patch

from autoconduck.agents.base import BaseAdapter, BEGIN_MARKER, END_MARKER


class _Adapter(BaseAdapter):
    id = "test-agent"

    def detect(self):
        return True

    def config_paths(self):
        return [self.config_path]

    def patch(self, config):
        return None


def _adapter(tmp_path):
    adapter = _Adapter()
    adapter.config_path = tmp_path / "agent.conf"
    return adapter


def test_upsert_block_writes_delimiters_and_preserves_content(monkeypatch):
    with tempfile.TemporaryDirectory() as home:
        monkeypatch.setenv("AUTOCONDUCK_HOME", home)
        path = Path(home) / "agent.conf"
        path.write_text("before\n", encoding="utf-8")
        adapter = _adapter(Path(home))

        adapter._upsert_block(path, "model = cheap")

        text = path.read_text(encoding="utf-8")
        assert "before\n" in text
        assert f"{BEGIN_MARKER}\nmodel = cheap\n{END_MARKER}" in text


def test_upsert_backs_up_before_write_and_prunes_to_five(monkeypatch):
    with tempfile.TemporaryDirectory() as home:
        monkeypatch.setenv("AUTOCONDUCK_HOME", home)
        path = Path(home) / "agent.conf"
        path.write_text("original\n", encoding="utf-8")
        adapter = _adapter(Path(home))

        timestamps = (f"20260101T00000{index}" for index in range(7))
        with patch("autoconduck.agents.base.time.strftime", side_effect=timestamps):
            for index in range(7):
                path.write_text(f"version-{index}\n", encoding="utf-8")
                adapter._upsert_block(path, f"value = {index}")

        backups = list((Path(home) / "backups" / adapter.id).glob("*.bak"))
        assert len(backups) == 5
        assert any(backup.read_text(encoding="utf-8") == "version-6\n" for backup in backups)


def test_revert_restores_original_content_from_backup(monkeypatch):
    with tempfile.TemporaryDirectory() as home:
        monkeypatch.setenv("AUTOCONDUCK_HOME", home)
        path = Path(home) / "agent.conf"
        original = "setting = keep\n"
        path.write_text(original, encoding="utf-8")
        adapter = _adapter(Path(home))

        adapter._upsert_block(path, "setting = patched")
        adapter.revert()

        assert path.read_text(encoding="utf-8") == original


def test_strip_block_removes_only_delimited_block(monkeypatch):
    with tempfile.TemporaryDirectory() as home:
        monkeypatch.setenv("AUTOCONDUCK_HOME", home)
        path = Path(home) / "agent.conf"
        path.write_text(
            f"keep-before\n{BEGIN_MARKER}\nmanaged\n{END_MARKER}\nkeep-after\n",
            encoding="utf-8",
        )
        adapter = _adapter(Path(home))

        adapter._strip_block(path)

        assert path.read_text(encoding="utf-8") == "keep-before\nkeep-after\n"
