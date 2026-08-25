import json
import subprocess
import sys

HEAVY = ("fastapi", "litellm", "onnxruntime", "textual")


def _loaded(code):
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    return json.loads(result.stderr.strip().rsplit("\n", 1)[-1])


def test_importing_main_is_lightweight():
    code = "import autoconduck.main, json, sys; print(json.dumps([n for n in %r if n in sys.modules]), file=sys.stderr)" % (HEAVY,)
    assert _loaded(code) == []


def test_help_and_version_are_lightweight():
    for option in ("--help", "--version"):
        code = "import autoconduck.main as m, json, sys\ntry: m.main(%r)\nexcept SystemExit: pass\nprint(json.dumps([n for n in %r if n in sys.modules]), file=sys.stderr)" % ((option,), HEAVY)
        assert _loaded(code) == []
