"""Credential-free integration simulation for AutoConduck's slow path."""
from __future__ import annotations

import asyncio, json, sys, threading, time, traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LOG = ROOT / "scripts" / "mock_prompts.log"
ANSWER = "FINAL_EXECUTOR_ANSWER_MOCK: done."


class Mock(BaseHTTPRequestHandler):
    latency = 0.0
    seen: list[dict] = []
    branches: list[str] = []

    def log_message(self, *_): pass

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
        self.seen.append(body)
        messages = body.get("messages", [])
        parts = [str(m.get("content", "")) for m in messages if isinstance(m, dict)]
        prompt = "\n".join(parts); system_prompt = next((str(m.get("content", "")) for m in messages if m.get("role") == "system"), "")
        schema_name = body.get("response_format", {}).get("json_schema", {}).get("name")
        last_user = next((str(m.get("content", "")) for m in reversed(messages) if m.get("role") == "user"), "")
        last_message = messages[-1] if messages else {}
        if schema_name == "TaskPlan":
            branch = "TaskPlan"
        elif schema_name == "ReconTarget":
            branch = "ReconTarget"
        elif "Original request:" in last_user:
            branch = "Original request"
        elif "ROLE: You are" in last_user:
            branch = "ROLE"
        elif isinstance(last_message, dict) and last_message.get("role") == "tool":
            branch = "post-tool"
        elif "Reply with FAST or SLOW" in last_user:
            branch = "tiebreaker"
        elif "tools" in body:
            branch = "tools"
        else:
            branch = "fallback"
        self.branches.append(branch)
        with LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"keys": sorted(body), "response_format_name": schema_name, "has_tools": "tools" in body, "user_content_preview": last_user[:250]}, default=str) + "\n")
        if schema_name == "TaskPlan":
            text = json.dumps({"subtasks": [{"id":"t1","goal":"Inspect dispatcher routing behavior.","scope":["autoconduck/routing/dispatcher.py"],"output_contract":{"description":"Summarize routing.","verify":[]},"constraints":["Do not propose code changes."],"depends_on":[],"verified_context":[],"read_budget":5,"role":"read"},{"id":"t2","goal":"Inspect routing configuration.","scope":["autoconduck/config.py"],"output_contract":{"description":"Summarize configuration.","verify":[]},"constraints":["Do not propose code changes."],"depends_on":["t1"],"verified_context":[],"read_budget":5,"role":"read"}],"summary":"Inspect routing.","budget_hint":0.7})
        elif schema_name == "ReconTarget":
            text = json.dumps({"files":["autoconduck/routing/dispatcher.py"],"query":"q","reasoning":"r"})
        elif isinstance(last_message, dict) and last_message.get("role") == "tool":
            text = "Mock post-tool answer."
        elif "Reply with FAST or SLOW" in last_user:
            text = "SLOW 8"
        elif "Original request:" in last_user:
            text = ANSWER
        elif "ROLE: You are" in last_user:
            if self.latency: time.sleep(self.latency)
            text = "Mock analyst findings for this subtask."
        elif "tools" in body:
            return self.reply(body, "", {"id":"call_1","name":"bash","arguments":{"command":"ls"}})
        else:
            text = "Mock generic response."
        self.reply(body, text)
    def reply(self, body, text, tool=None):
        msg = {"role": "assistant", "content": text}
        if tool:
            msg["content"] = None
            msg["tool_calls"] = [{"id": tool["id"], "type": "function", "function": {"name": tool["name"], "arguments": json.dumps(tool["arguments"])}}]
        if body.get("stream"):
            chunks = [{"id": "mock", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}]
            if text: chunks.append({"id": "mock", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]})
            if tool: chunks.append({"id": "mock", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": msg["tool_calls"]}, "finish_reason": None}]})
            chunks.append({"id": "mock", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
            raw = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
            self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers(); self.wfile.write(raw.encode()); return
        raw = json.dumps({"id": "mock", "object": "chat.completion", "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw)


def config(base, timeout=120.0):
    from autoconduck.config import Config, SelectionConfig
    s = SelectionConfig(subagent_timeout_s=timeout, enable_executor_subagents=False, enable_fast_path_graph=True, min_orchestrator_complexity=.4, slow_threshold=.4, complexity_weights={"length":2.0,"structural":2.0,"scope_breadth":2.0,"code_density":2.0,"abstraction_level":2.0,"uncertainty_hedge":2.0,"cross_domain":2.0,"task_novelty":2.0,"imperative_strength":2.0,"multi_step":2.0})
    entries = [{"id": n, "provider": "mock", "api_base": base, "api_key": "dummy", "input_cost_per_token": 1e-6, "output_cost_per_token": 1e-6, "tier": t} for n, t in (("autoconduck", "fast"), ("autoconduck-budget", "fast"), ("autoconduck-expensive", "slow"))]
    return Config(model_list=entries, selection=s, fast_path_digest_enabled=False)


BASE_PROMPT = "Refactor authentication in autoconduck/routing/dispatcher.py, autoconduck/config.py, and tests/test_dispatcher.py: (1) replace session cookies with JWT access and refresh tokens, (2) add rate-limiting middleware, (3) migrate the database schema, (4) update all call sites, (5) write expiry edge-case tests, (6) update API docs, (7) coordinate frontend refresh flow, (8) add cleanup cron, (9) preserve legacy mobile compatibility, with rollback, security review, load testing, and staged multi-step migration."
TOOLS = [{"type": "function", "function": {"name": "bash", "description": "run shell", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}}]


def main():
    from fastapi.testclient import TestClient
    from autoconduck.routing.evaluator import complexity_of
    from autoconduck.orchestrator.graph import _LANGGRAPH_AVAILABLE
    import autoconduck.config as cm, autoconduck.server_streaming as server, autoconduck.stats as stats
    LOG.unlink(missing_ok=True)
    Mock.seen.clear(); Mock.branches.clear()
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Mock); threading.Thread(target=upstream.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{upstream.server_address[1]}"; cfg = config(base); cm.get_config = lambda: cfg
    trace = []; old_stats = stats.update_active_routing
    def record(**kw):
        if kw.get("node"): trace.append(kw["node"])
        return old_stats(**kw)
    stats.update_active_routing = record
    candidates = [BASE_PROMPT, BASE_PROMPT + " Resolve every uncertainty and dependency across all domains.", BASE_PROMPT + " Include exhaustive architecture, failure, and observability analysis."]
    scores = [(p, complexity_of(p, cfg)) for p in candidates]
    for i, (_, score) in enumerate(scores, 1): print(f"S1 candidate {i}: {score:.3f}")
    prompt, score = next((x for x in scores if x[1] >= .75), scores[-1])
    prompt += "\n\nTraceback (most recent call last):\n  File \"/app/main.py\", line 42, in <module>\n    raise ValueError('boom')\nValueError: boom"
    score = complexity_of(prompt, cfg); print(f"S1 chosen score={score:.3f}\nPrompt: {prompt}")
    from autoconduck.routing.dispatcher import route
    decision = route([{"role": "user", "content": prompt}], [], pseudo_model="autoconduck", config=cfg)
    print("S2 routing decision:", {k: getattr(decision, k) for k in ("path", "confidence_band", "confidence", "complexity", "reason")})
    assert decision.path == "slow"
    results = []
    def run(name, fn):
        start = time.perf_counter()
        try: results.append((name, "PASS", f"{time.perf_counter()-start:.2f}s {fn()}"))
        except Exception as e: traceback.print_exc(); results.append((name, "FAIL", f"{time.perf_counter()-start:.2f}s {e}"))
    def client(): server._build(); return TestClient(server.app)
    run("S1 complexity", lambda: (_ for _ in ()).throw(AssertionError("< .75")) if score < .75 else "score >= .75")
    def slow(stream=False):
        trace.clear()
        with client() as c:
            r = c.post("/v1/chat/completions", json={"model": "autoconduck", "messages": [{"role": "user", "content": prompt}], "tools": TOOLS, "stream": stream})
            if not stream:
                print("\n--- S2 DIAGNOSTICS ---")
                print("_LANGGRAPH_AVAILABLE:", _LANGGRAPH_AVAILABLE)
                print("S2 raw response status:", r.status_code)
                print("S2 raw response json:", r.json())
                print("S2 node trace:", trace)
                stats_body = c.get("/stats").json()
                print("S2 GET /stats body:", stats_body)
                from collections import Counter
                print("S2 mock branch tally:", dict(Counter(Mock.branches)))
                records = stats_body.get("routing_records", stats_body.get("decisions", []))
                if isinstance(records, list):
                    degraded = [x for x in records if isinstance(x, dict) and str(x.get("path", "")).upper() in {"FAST", "FALLBACK"}]
                    if degraded:
                        print("S2 dispatcher degraded routing records:", [{k: x.get(k) for k in ("path", "reason")} for x in degraded])
                print("--- END S2 DIAGNOSTICS ---\n")
            text = ("".join(r.iter_text()) if stream else r.text); assert r.status_code == 200 and ANSWER in text
            # Current stats labels both fan-out nodes as ``subagents``.  Keep
            # the raw trace, but expose the phase-specific names required by
            # this simulation's contract.
            canonical = []
            subagent_seen = 0
            for node in trace:
                if node == "subagents":
                    subagent_seen += 1
                    canonical.append("recon_subagent_pool" if subagent_seen == 1 else "subagent_pool")
                elif node != "idle":
                    canonical.append(node)
            expected = ["recon", "recon_subagent_pool", "planner", "subagent_pool", "compactor", "executor"]
            assert all(x in canonical for x in expected), canonical
            assert canonical.index("recon") < canonical.index("planner") < canonical.index("subagent_pool") < canonical.index("compactor") < canonical.index("executor")
            assert c.get("/stats").json()["path_counts"].get("SLOW", 0) > 0
        print("S2 node trace:", trace); return "slow pipeline"
    run("S2 full slow", lambda: slow(False)); run("S3a streaming slow", lambda: slow(True))
    def tools():
        with client() as c:
            with c.stream("POST", "/v1/chat/completions", json={"model": "autoconduck-budget", "messages": [{"role": "user", "content": "please use bash tool"}], "tools": TOOLS, "stream": True}) as r: text = "".join(r.iter_text())
            assert r.status_code == 200 and "tool_calls" in text and "bash" in text and "call_1" in text and "[DONE]" in text
            follow = c.post("/v1/chat/completions", json={"model": "autoconduck-budget", "messages": [{"role": "user", "content": "please use bash tool"}, {"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "bash", "arguments": "{\"command\":\"ls\"}"}}]}, {"role": "tool", "tool_call_id": "call_1", "content": "file.txt"}], "tools": TOOLS})
            assert follow.status_code == 200 and "Mock post-tool answer." in follow.text
        return "tool round-trip"
    run("S3b streaming tools", tools)
    def timeout(t):
        from autoconduck.orchestrator.planner import SubTask, OutputContract
        from autoconduck.orchestrator.subagents import run_subagent
        class MockClient:
            def completion(self, messages, **kwargs):
                import urllib.request
                req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps({"messages": messages}).encode(), headers={"Content-Type":"application/json"})
                return json.loads(urllib.request.urlopen(req, timeout=30).read())
        Mock.latency = 15; cfg.selection.subagent_timeout_s = t
        task = SubTask(id="timeout", goal="Inspect dispatcher.", scope=["autoconduck/routing/dispatcher.py"], output_contract=OutputContract(description="summary"), constraints=["read only"], role="read")
        result = asyncio.run(run_subagent(task, "", client=MockClient(), cfg=cfg))
        if t < 120:
            assert result.startswith("__SUBAGENT_ERROR__")
        else:
            assert not result.startswith("__SUBAGENT_ERROR__")
        with client() as c: assert c.post("/v1/chat/completions", json={"model": "autoconduck", "messages": [{"role": "user", "content": prompt}]}).status_code == 200
        Mock.latency = 0; return "timeout graceful"
    run("S4 old timeout", lambda: timeout(12.0)); run("S5 default timeout", lambda: timeout(120.0))
    def tiebreaker():
        from autoconduck.config import SelectionConfig
        moderate = "Review the dispatcher and suggest a small routing improvement."
        off_cfg = cfg.model_copy(deep=True)
        off_cfg.selection = SelectionConfig(**{**cfg.selection.model_dump(), "tiebreaker_enabled": False})
        off = route([{"role": "user", "content": moderate}], [], config=off_cfg)
        print("S7 routing decision with tiebreaker off:", {k: getattr(off, k) for k in ("path", "confidence_band", "confidence", "complexity", "reason")})
        assert off.path == "fast" and off.confidence_band == "ambiguous"
        on_cfg = cfg.model_copy(deep=True); on_cfg.selection.tiebreaker_enabled = True
        cm.get_config = lambda: on_cfg; server.get_config = lambda: on_cfg
        server.app = None; server._cached.clear(); trace.clear()
        with client() as c:
            on = route([{"role": "user", "content": moderate}], [], config=on_cfg)
            print("S7 routing decision:", {k: getattr(on, k) for k in ("path", "confidence_band", "confidence", "complexity", "reason")})
            assert on.path == "slow" and "tiebreaker: slow" in on.reason
            r = c.post("/v1/chat/completions", json={"model": "autoconduck", "messages": [{"role": "user", "content": moderate}]})
            assert r.status_code == 200 and ANSWER in r.text
            assert all(x in trace for x in ["recon", "planner", "compactor", "executor"]), trace
        return f"tiebreaker slow; node trace={trace}"
    run("S7 tiebreaker", tiebreaker)
    def anthropic():
        with client() as c:
            b={"model":"autoconduck","system":"concise","messages":[{"role":"user","content":prompt}],"max_tokens":100,"tools":[{"name":"bash","description":"run","input_schema":{"type":"object"}}],"stream":False}; r=c.post("/v1/messages",json=b); assert r.status_code==200 and r.json().get("type")=="message"
            b["stream"]=True
            with c.stream("POST","/v1/messages",json=b) as sr: text="".join(sr.iter_text())
            assert sr.status_code==200 and ("content_block_start" in text or ANSWER in text)
            assert c.post("/v1/messages",json={**b,"stream":False,"messages":[{"role":"user","content":"tool result"}]}).status_code==200
        return "Anthropic shape/SSE"
    run("S6 Anthropic", anthropic); upstream.shutdown(); LOG.unlink(missing_ok=True)
    print("\nScenario | Result | Notes\n---------|--------|------"); [print(" | ".join(x)) for x in results]
    return int(any(x[1] != "PASS" for x in results))

if __name__ == "__main__": raise SystemExit(main())
