"""Pure SLM download progress state helpers."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DownloadProgress:
    downloaded: int = 0
    total: int = 0
    status: str = "pending"

    @property
    def percent(self) -> int | None:
        if self.total <= 0:
            return None
        return min(100, max(0, int(self.downloaded * 100 / self.total)))


def download_progress(downloaded: int, total: int) -> DownloadProgress:
    """Normalize downloader callback values into renderable progress state."""
    downloaded = max(0, int(downloaded))
    total = max(0, int(total))
    status = "complete" if total > 0 and downloaded >= total else "downloading"
    return DownloadProgress(downloaded=downloaded, total=total, status=status)
