from dataclasses import dataclass


@dataclass
class ProgressEvent:
    kind: str
    name: str
    state: str
    detail: str = ""
    index: int = 0
    total: int = 0
    elapsed_s: float = 0.0


class ProgressFormatter:
    def __init__(self, config=None):
        self.config = config

    def format(self, ev: ProgressEvent) -> str | None:
        verbosity = getattr(getattr(self.config, "selection", self.config), "progress_verbosity", "verbose")
        if verbosity == "off":
            return None
        if verbosity == "terse":
            return ev.name if ev.state == "running" else None
        if ev.kind == "pool" and ev.state == "running":
            return f"⏳ {ev.total} subagents"
        if ev.kind == "footer":
            return "✓ Workflow completed."
        name = ev.name
        if ev.kind == "subagent" and ev.total:
            name = f"subagent {ev.index}/{ev.total} ({ev.detail.split(' · ', 1)[0]})"
            detail = ev.detail.split(" · ", 1)[-1] if " · " in ev.detail else ""
        else:
            detail = ev.detail
        glyph = "●" if ev.state == "running" else "✓" if ev.state == "done" else "✗"
        suffix = f" · {detail}" if detail else (f" · {ev.elapsed_s:.1f}s" if ev.state == "done" else "")
        return f"{glyph} {name} · {ev.state}{suffix}"
