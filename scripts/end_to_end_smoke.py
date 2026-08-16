"""Bounded manual smoke test for a configured AutoConduck gateway."""
import argparse
import json
from fastapi.testclient import TestClient

from autoconduck.main import _build

CASES = [
    ("fast", "autoconduck", "fix this typo", 16),
    ("budget", "autoconduck-budget", "fix this typo", 16),
    ("expensive", "autoconduck-expensive", "fix this typo", 16),
    ("slow", "autoconduck", "Refactor the application: audit the fast-path latency budget, review the architecture, write integration tests for the whole system, propose and apply optimizations across multiple files, and add regression tests.", 1024),
    ("messages", "autoconduck", "fix this typo", 16),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        for name, model, prompt, tokens in CASES:
            print(f"would call {name}: {model}, max_tokens={tokens}, prompt={prompt}")
        print("would call models and stats endpoints")
        return
    _build()
    import autoconduck.main as main_module
    rows = []
    with TestClient(main_module.app) as client:
        for name, model, prompt, tokens in CASES:
            try:
                if name == "messages":
                    response = client.post("/v1/messages", json={"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": tokens, "stream": False})
                    data = response.json(); output = ((data.get("content") or [{}])[0].get("text", ""))
                else:
                    response = client.post("/v1/chat/completions", json={"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": tokens})
                    data = response.json(); output = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                response_model = data.get("model") or "orchestrator-answer (no model field)"
                rows.append((name, model, response_model, response.status_code, str(output)[:80]))
            except Exception as exc:
                rows.append((name, model, "error", "ERR", str(exc)[:80]))
        for endpoint in ("/v1/models", "/stats"):
            try:
                response = client.get(endpoint)
                rows.append((endpoint, "-", "-", response.status_code, response.text[:80]))
            except Exception as exc:
                rows.append((endpoint, "-", "error", "ERR", str(exc)[:80]))
    print("mode\trequested\tresponse model\tstatus\toutput")
    for row in rows:
        print("\t".join(map(str, row)))
    print("Estimated cost: inspect /stats; this smoke uses at most five tiny model calls plus two local endpoints.")


if __name__ == "__main__":
    main()
