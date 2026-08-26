"""Measure cold CLI startup and best-effort TUI construction."""
import statistics
import subprocess
import sys
import time

HEAVY = ("fastapi", "litellm", "onnxruntime", "textual")


def measure(option, runs=8):
    values, imported = [], set()
    for _ in range(runs):
        code = "import autoconduck.main as m, sys\ntry: m.main(%r)\nexcept SystemExit: pass\nprint('IMPORTED', [n for n in %r if n in sys.modules], file=sys.stderr)" % ((option,), HEAVY)
        started = time.perf_counter()
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        values.append((time.perf_counter() - started) * 1000)
        imported.update(result.stderr.rsplit("IMPORTED ", 1)[-1].strip("[]\n ").replace("'", "").split(", "))
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, int(len(ordered) * .95))]
    print(f"{option}: mean={statistics.mean(values):.1f}ms p50={statistics.median(values):.1f}ms p95={p95:.1f}ms")
    return max(values), imported


def main():
    worst, imported = 0, set()
    for option in ("--version", "--help"):
        elapsed, modules = measure(option)
        worst, imported = max(worst, elapsed), imported | modules
    print(f"Heavy modules observed during CLI runs: {sorted(x for x in imported if x)}")
    try:
        from autoconduck.tui.app import AutoConduckApp
        started = time.perf_counter()
        AutoConduckApp(configured=False)
        print(f"TUI app construction: {(time.perf_counter() - started) * 1000:.1f}ms (frame requires event loop)")
    except (ImportError, RuntimeError) as exc:
        print(f"TUI measurement skipped: {exc}")
    return 1 if worst >= 1000 else 0


if __name__ == "__main__":
    raise SystemExit(main())
