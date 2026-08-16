"""Probe configured models to distinguish model JSON quality from orchestration."""
import sys
import time

from autoconduck.config import get_config
from autoconduck.jsonutil import parse_json_text
from autoconduck.messages_api import litellm_params_for
from autoconduck import pricing


def main() -> int:
    cfg = get_config()
    requested = set(sys.argv[1:])
    models = pricing.pool_ids(cfg)
    if requested:
        models = [model for model in models if model in requested or model.split("/", 1)[-1] in requested]
    if not models:
        print("Could not run live: no configured pool models. Configure ~/.autoconduck/config.yaml and auth.yaml.")
        return 0
    import litellm
    schema = {"type": "object", "properties": {"task": {"type": "string"}, "steps": {"type": "array", "items": {"type": "string"}}, "depth": {"type": "integer"}}, "required": ["task", "steps", "depth"], "additionalProperties": False}
    prompt = 'Return JSON only: {"task":"summarize auth flow","steps":["read","plan","write"],"depth":3}'
    rows = []
    for model in models:
        for mode, fmt in (("plain", None), ("json_object", {"type": "json_object"}), ("json_schema", {"type": "json_schema", "json_schema": {"name": "Probe", "schema": schema, "strict": True}})):
            started = time.perf_counter()
            try:
                params = {**litellm_params_for(model, cfg), "model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 200, "temperature": 0.0, "timeout": 30}
                if fmt:
                    params["response_format"] = fmt
                response = litellm.completion(**params)
                raw = response.choices[0].message.content or ""
                if not raw.strip(): status = "EMPTY"
                else:
                    parsed, repair, preview = parse_json_text(raw)
                    status = "VALID" if repair is None and parsed is not None else "INVALID-REPAIRABLE" if parsed is not None else "INVALID"
                rows.append((model, mode, status, time.perf_counter() - started, raw[:80]))
            except Exception as exc:
                rows.append((model, mode, "ERROR", time.perf_counter() - started, str(exc)[:80]))
    print("model | mode | status | latency | preview")
    for model, mode, status, latency, preview in rows:
        print(f"{model} | {mode} | {status} | {latency:.2f}s | {preview.replace(chr(10), ' ')}")
    for model in models:
        subset = [row for row in rows if row[0] == model]
        print(f"{model}: " + "; ".join(f"{row[1]} -> {row[2]}" for row in subset))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
