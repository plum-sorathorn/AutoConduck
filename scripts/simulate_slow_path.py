"""Credential-free integration simulation for AutoConduck's slow path."""
from __future__ import annotations

import asyncio, json, os, sys, threading, time, traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Progress deltas contain unicode glyphs (● ✓ ⏳ ·); Windows consoles often
# default stdout to cp1252, which can't encode them. Force utf-8 so printing
# real progress transcripts doesn't crash the simulation.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LOG = ROOT / "scripts" / "mock_prompts.log"
ANSWER = "FINAL_EXECUTOR_ANSWER_MOCK: done."


def _sse_content(raw_sse_text: str) -> str:
    """Decode raw ``data: {...}\n\n`` SSE frames into the assembled delta
    content text. Needed because json.dumps(ensure_ascii=True) escapes the
    progress glyphs (⠋ ✓ ⏳) as literal ``\\uXXXX`` sequences in the wire
    text, so substring checks against real unicode glyphs must run against
    the *decoded* content, not the raw SSE bytes."""
    parts = []
    for line in raw_sse_text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        if payload.strip() == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except Exception:
            continue
        for choice in obj.get("choices", []):
            content = (choice.get("delta") or {}).get("content")
            if content:
                parts.append(content)
    return "".join(parts)


class Mock(BaseHTTPRequestHandler):
    latency = 0.0
    seen: list[dict] = []
    branches: list[str] = []
    planner_calls = 0
    planner_failure = False
    planner_truncation = False
    planner_retry = False
    role_plan = False

    def log_message(self, *_): pass

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
        self.seen.append(body)
        if self.latency:
            time.sleep(self.latency)
        messages = body.get("messages", [])
        parts = [str(m.get("content", "")) for m in messages if isinstance(m, dict)]
        prompt = "\n".join(parts); system_prompt = next((str(m.get("content", "")) for m in messages if m.get("role") == "system"), "")
        schema_name = body.get("response_format", {}).get("json_schema", {}).get("name")
        response_format_type = body.get("response_format", {}).get("type")
        is_planner = any(
            "You are a coding-task planner" in str(m.get("content", ""))
            for m in messages
            if isinstance(m, dict)
        )
        is_recon = any(
            "You are a codebase reconnaissance assistant" in str(m.get("content", ""))
            for m in messages
            if isinstance(m, dict)
        )
        last_user = next((str(m.get("content", "")) for m in reversed(messages) if m.get("role") == "user"), "")
        last_message = messages[-1] if messages else {}
        response_format_type = body.get("response_format", {}).get("type")
        if is_planner:
            branch = "TaskPlan"
        elif is_recon:
            branch = "ReconTarget"
        elif "Original request:" in last_user:
            branch = "Original request"
        elif "You are a coding-task planner" in system_prompt:
            branch = "TaskPlan"
        elif "ROLE: You are" in last_user:
            branch = "ROLE"
        elif isinstance(last_message, dict) and last_message.get("role") == "tool":
            branch = "post-tool"
        elif last_user.startswith("Reply with FAST or SLOW"):
            branch = "tiebreaker"
        elif "tools" in body:
            branch = "tools"
        else:
            branch = "fallback"
        self.branches.append(branch)
        with LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"keys": sorted(body), "response_format_name": schema_name, "has_tools": "tools" in body, "user_content_preview": last_user[:250]}, default=str) + "\n")
        if is_planner:
            self.__class__.planner_calls += 1
            if self.__class__.planner_failure:
                text = ""
                self.reply(body, text)
                return
            if self.__class__.role_plan:
                clean_plan = json.dumps({"subtasks": [
                    {"id":"review-auth","goal":"Review the auth flow changes for regressions","scope":["autoconduck/config.py"],"output_contract":{"description":"Review findings.","verify":[]},"constraints":["Do not edit."],"depends_on":[],"verified_context":[],"read_budget":5,"role":"read"},
                    {"id":"logging-helper","goal":"Implement a logging helper","scope":["autoconduck/config.py"],"output_contract":{"description":"Implementation draft.","verify":[]},"constraints":["Stay in scope."],"depends_on":[],"verified_context":[],"read_budget":5,"role":"write"}],"summary":"Role test.","budget_hint":0.7})
            else:
                clean_plan = json.dumps({"subtasks": [{"id":"t1","goal":"Inspect dispatcher routing behavior.","scope":["autoconduck/routing/dispatcher.py"],"output_contract":{"description":"Summarize routing.","verify":[]},"constraints":["Do not propose code changes."],"depends_on":[],"verified_context":[],"read_budget":5,"role":"read"}],"summary":"Inspect routing.","budget_hint":0.7})
            if self.__class__.planner_retry and self.__class__.planner_calls == 1:
                text = "{broken planner json"
                self.reply(body, text)
                return
            if self.__class__.planner_truncation and response_format_type is None:
                text = '{"subtasks":[{"id":"t1","goal":"Inspect auth'
            elif schema_name is None and response_format_type is None:
                text = clean_plan
            elif response_format_type == "json_object":
                text = clean_plan
            else:
                text = '```json\n{"subtasks":[{"id":"t1"'
        elif is_recon:
            text = json.dumps({"files":["autoconduck/routing/dispatcher.py"],"query":"q","reasoning":"r"})
        elif isinstance(last_message, dict) and last_message.get("role") == "tool":
            text = "Mock post-tool answer."
        elif last_user.startswith("Reply with FAST or SLOW"):
            text = "SLOW 8"
        elif "Original request:" in last_user:
            text = ANSWER
        elif "ROLE: You are" in last_user or "subagent" in last_user:
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


def config(base, timeout=120.0, slow_stream_progress=True):
    from autoconduck.config import Config, SelectionConfig
    s = SelectionConfig(subagent_timeout_s=timeout, enable_executor_subagents=False, enable_fast_path_graph=True, min_orchestrator_complexity=.4, slow_threshold=.75, ambiguous_low=.40, slow_stream_progress=slow_stream_progress)
    # NOTE: pricing.py's _entry()/_configured_entry() lookup keys off
    # ``price_in``/``price_out`` (USD per million tokens), matching
    # config.ModelEntry's schema -- NOT the litellm-style
    # ``input_cost_per_token``/``output_cost_per_token`` keys. Using the
    # wrong key here silently zeroes every entry's effective cost, making
    # cheapest_enabled() tie-break alphabetically instead of by price.
    entries = [{"id": n, "provider": "mock", "api_base": base, "api_key": "dummy", "price_in": cost, "price_out": cost, "tier": t} for n, t, cost in (("autoconduck", "fast", 1.0), ("autoconduck-budget", "fast", 1.0), ("autoconduck-expensive", "slow", 1.0), ("mock-cheapest", "fast", 1e-6))]
    return Config(model_list=entries, selection=s, fast_path_digest_enabled=False)


BASE_PROMPT = "Refactor the authentication module in src/auth.py and src/utils.py: (1) replace session cookies with JWT access+refresh tokens, (2) add rate-limiting middleware..., (3) migrate the user table schema..., (4) update all 14 call sites..., (5) write unit tests covering token expiry edge cases, (6) update the API documentation, (7) coordinate with the frontend team on the refresh flow, (8) add a cleanup cron job for expired tokens, (9) preserve backward compatibility with the legacy mobile client..."
TOOLS = [{"type": "function", "function": {"name": "bash", "description": "run shell", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}}]


S9_PROMPT = """
This is a large, multi-part review, audit, and remediation task across the entire
AutoConduck codebase. The scope of this task is complex and additional planning and decomposition are required before any implementation begins. Treat it as a full architecture-level overhaul, not a list
of quick fixes:

1. Custom provider page (onboarding): the placeholder text for the OpenAI base
   URL is wrong — it currently says "localhost" but should show a real
   gateway-style URL such as https://opencode.ai/zen/go/v1 (example only; do not
   hard-code that exact URL — derive it per provider).

2. TUI main-menu rendering glitch: random stray lines are drawn on the main menu
   screen of the AutoConduck TUI. Investigate the root cause (looks like a
   rendering dirty-region bug) and remove them, or render them properly. While
   investigating, we hit this traceback when opening the dashboard:

   Traceback (most recent call last):
     File "autoconduck/tui/dashboard.py", line 222, in on_mount
       self._render_menu()
   AttributeError: 'Dashboard' object has no attribute '_layout_rows'

3. Post-onboarding navigation: after a successful onboarding/configuration run,
   the user should be sent back to the main menu — not the live routing stats
   page. Make this the case for all other sub-pages too (consistent return-to-
   menu flow).

4. Settings page interactions: (a) for toggleable True/False settings, pressing
   Enter on a selected setting should flip the value directly — users must not
   type "True"/"False"; (b) numerical decimal settings must accept only
   increments of 0.05 and be capped at their documented maximum; (c) Log Level
   should also toggle/cycle with Enter instead of free-text entry.

5. Simple Tuning page: remove the headroom % field; the usd/tokens input should
   be a toggle; keybinds must always be visible; input hints should display
   separately; going back to the main menu with "b" is too hard — use Esc, and
   support Esc + left-arrow on every page (except text-input pages, where only
   Esc applies).

6. Live routing stats page: must expose the same information density as the
   "conduck stats" command.

7. Advanced tuning mode: arrow keys currently do nothing — the keybind
   configuration is almost certainly wired incorrectly. Fix it and show the user
   which parameter is currently selected.

8. Keybinding consistency: never use "b" to go back; always Esc. Unify keybinds
   across all pages.

9. Startup latency optimization: launching a coding agent from AutoConduck takes
   too long, and `conduck start --headless` itself is slow. Profile the startup
   pipeline end-to-end (lazy imports, model-catalog ingestion, provider warm-up)
   and eliminate critical bottlenecks — especially I/O and heavy imports on the
   hot path. If no significant optimization is possible, add a loading screen
   immediately after the start command so the user knows it isn't frozen.
   Examine concurrency and latency around process spawn and the failover
   watchdog.

10. Visual glitch: AutoConduck stays on screen after a coding agent is launched
    from the launch menu — the process handoff/rollback path leaks the TUI.
    Diagnose and fix.

11. Preset model selection defaults: some models on some presets ship
    pre-selected. They should all default to deselected, except presets with 6
    models or fewer, which should have all selected. This may be a
    migration/schema issue in the preset data.

12. uninstall / update lifecycle: `autoconduck uninstall` and `autoconduck
    update` must also kill the running AutoConduck process if one exists, and
    must run in the foreground terminal (no background process spawning) so the
    user sees completion.

13. First-run onboarding install latency: installing AutoConduck onto a coding
    agent takes too long on first onboarding. Optimize it and/or show a loading
    screen before installation begins so users know it isn't frozen.

14. At any point in time EXCEPT planning, autoconduck should be able to deploy subagents even in fast path. However, we need to make it actually useful, faster than a single model execution (minimal but accurate planning in fast path), not have merge/conflict issues like imports not having the same name or code implemented with the same idea but does not work together etc.

15. command "conduck" doesn't actually start the proxy but it says proxy active on the top. It needs to work the same as conduck start, and leads us the to the main menu.

16. Release: bump the version number to 0.2.1.

This task spans multiple subsystems — TUI/keybinding architecture, FastAPI
server startup, launcher/process lifecycle, preset data schema, routing
configuration, and packaging — so a full plan with sub-agent decomposition, and
a verification pass with tests for every change, is required.
"""


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
    plain_prompt = BASE_PROMPT
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
    def slow(stream=False, pipeline_prompt=prompt, use_tools=True):
        trace.clear()
        with client() as c:
            payload = {"model": "autoconduck", "messages": [{"role": "user", "content": pipeline_prompt}], "stream": stream}
            if use_tools:
                payload["tools"] = TOOLS
            r = c.post("/v1/chat/completions", json=payload)
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
    run("S2 full slow", lambda: slow(False))
    def plain_complex_slow():
        s8_cfg = cfg.model_copy(deep=True)
        s8_cfg.selection.complexity_weights = {"scope_breadth": .4, "imperative_strength": .3, "cross_domain": .3}
        cm.get_config = lambda: s8_cfg; server.get_config = lambda: s8_cfg
        server.app = None; server._cached.clear()
        plain_score = complexity_of(plain_prompt, s8_cfg)
        plain_decision = route([{"role": "user", "content": plain_prompt}], [], pseudo_model="autoconduck", config=s8_cfg)
        print(f"S8 plain prompt complexity={plain_score:.3f}")
        print("S8 routing decision:", {k: getattr(plain_decision, k) for k in ("path", "confidence_band", "confidence", "complexity", "reason")})
        assert plain_score >= .75, plain_score
        assert plain_decision.path == "slow"
        assert plain_decision.reason == "complexity threshold"
        return slow(False, plain_prompt, use_tools=False)
    run("S8 plain complex slow", plain_complex_slow)
    def streaming_opt_out():
        nonlocal cfg
        opt_out_cfg = cfg.model_copy(deep=True)
        opt_out_cfg.selection.slow_stream_progress = False
        cm.get_config = lambda: opt_out_cfg; server.get_config = lambda: opt_out_cfg
        server.app = None; server._cached.clear()
        with client() as c:
            with c.stream("POST", "/v1/chat/completions", json={"model": "autoconduck", "messages": [{"role": "user", "content": prompt}], "stream": True}) as response:
                text = _sse_content("".join(response.iter_text()))
        assert response.status_code == 200 and ANSWER in text
        assert not any(f"{label} ·" in text for label in ("recon", "planner", "subagent", "executor")), text
        cfg = cfg.model_copy(deep=True)
        cm.get_config = lambda: cfg; server.get_config = lambda: cfg
        server.app = None; server._cached.clear()
        return "explicit config opt-out has no progress lines"
    run("S3a streaming opt-out", streaming_opt_out)

    def streaming_default_on():
        # Rebuild from a fresh default config so this proves the default, not
        # state inherited from the explicit opt-out scenario.
        nonlocal cfg
        cfg = config(base)
        cm.get_config = lambda: cfg; server.get_config = lambda: cfg
        server.app = None; server._cached.clear()
        with client() as c:
            with c.stream("POST", "/v1/chat/completions", json={"model": "autoconduck", "messages": [{"role": "user", "content": prompt}], "stream": True}) as response:
                text = _sse_content("".join(response.iter_text()))
        assert response.status_code == 200 and ANSWER in text
        # Progress is part of the default stream contract; verify it precedes
        # the answer. Real deltas are formatted by ProgressFormatter
        # (progress.py) as e.g. "● recon · running · ..." -- not the legacy
        # bracketed "[recon]" strings, which only applied to the now-unused
        # dict-shaped event branch. Anchor on the leading glyph+space so
        # "subagent_pool" doesn't spuriously match inside
        # "recon_subagent_pool" (a substring collision).
        node_names = ("recon", "recon_subagent_pool", "planner", "subagent_pool", "compactor", "executor")
        needles = [f"● {name} \u00b7" for name in node_names]
        positions = [text.index(n) for n in needles if n in text]
        assert positions == sorted(positions) and positions
        assert positions == sorted(positions) and all(position < text.index(ANSWER) for position in positions), text
        return "default-on progress deltas precede final answer"
    run("S10 default streaming progress", streaming_default_on)

    def streaming_env_opt_out():
        old = os.environ.get("AUTOCONDUCK_STREAM_PROGRESS")
        os.environ["AUTOCONDUCK_STREAM_PROGRESS"] = "0"
        try:
            with client() as c:
                with c.stream("POST", "/v1/chat/completions", json={"model": "autoconduck", "messages": [{"role": "user", "content": prompt}], "stream": True}) as response:
                    text = _sse_content("".join(response.iter_text()))
            assert response.status_code == 200
            assert not any(f"{label} ·" in text for label in ("recon", "planner", "subagent", "executor")), text
        finally:
            if old is None:
                os.environ.pop("AUTOCONDUCK_STREAM_PROGRESS", None)
            else:
                os.environ["AUTOCONDUCK_STREAM_PROGRESS"] = old
        return "env 0 overrides default-on config"
    run("S11 env streaming opt-out", streaming_env_opt_out)
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
        object.__setattr__(cfg, "ambiguous_low", 0.40)
        off_cfg = cfg.model_copy(deep=True)
        off_cfg.selection = SelectionConfig(**{**cfg.selection.model_dump(), "tiebreaker_enabled": False})
        candidates = [
            "Reply with FAST or SLOW: " + BASE_PROMPT + " Include dependency, uncertainty, tradeoff, and validation analysis.",
            "Reply with FAST or SLOW: " + BASE_PROMPT + " Include architecture, integration, compatibility, failure-mode, and staged verification analysis.",
            "Reply with FAST or SLOW: " + BASE_PROMPT + " Include architecture analysis across related modules and interfaces, with rollback and observability.",
            "Reply with FAST or SLOW: " + BASE_PROMPT + " Include architecture analysis across authentication, API, and persistence boundaries, with staged verification.",
        ]
        scored = []
        for index, candidate in enumerate(candidates, 1):
            candidate_score = complexity_of(candidate, off_cfg)
            candidate_decision = route([{"role": "user", "content": candidate}], [], config=off_cfg)
            scored.append((candidate, candidate_score, candidate_decision))
            print(f"S7 candidate {index}: complexity={candidate_score:.3f} path={candidate_decision.path} band={candidate_decision.confidence_band}")
        moderate, score, off = next(
            item for item in scored
            if 0.55 <= item[1] <= 0.70 and item[2].path == "fast" and item[2].confidence_band == "ambiguous"
        )
        print(f"S7 chosen prompt complexity={score:.3f}: {moderate}")
        print("S7 routing decision with tiebreaker off:", {k: getattr(off, k) for k in ("path", "confidence_band", "confidence", "complexity", "reason")})
        assert off.path == "fast" and off.confidence_band == "ambiguous"
        on_cfg = cfg.model_copy(deep=True); on_cfg.selection.tiebreaker_enabled = True
        cm.get_config = lambda: on_cfg; server.get_config = lambda: on_cfg
        server.app = None; server._cached.clear(); trace.clear()
        with client() as c:
            on = route([{"role": "user", "content": moderate}], [], config=on_cfg)
            print("S7 routing decision with tiebreaker on:", {k: getattr(on, k) for k in ("path", "confidence_band", "confidence", "complexity", "reason")})
            assert on.path == "slow" and on.reason == "tiebreaker: slow"
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
    def s9_measure():
        nonlocal cfg
        cm.get_config = lambda: cfg; server.get_config = lambda: cfg
        server.app = None; server._cached.clear()
        s9_score = complexity_of(S9_PROMPT, cfg)
        s9_decision = route([{"role": "user", "content": S9_PROMPT}], [], pseudo_model="autoconduck", config=cfg)
        s9_fields = ("path", "confidence_band", "confidence", "complexity", "reason")
        print(f"S9 complexity: {s9_score:.3f}")
        print("S9 routing decision:", {k: getattr(s9_decision, k) for k in s9_fields})
        assert s9_decision.path == "slow"

        def execute(label, latency):
            trace.clear()
            Mock.latency = latency
            before = len(Mock.seen)
            started = time.perf_counter()
            try:
                with client() as c:
                    response = c.post("/v1/chat/completions", json={
                        "model": "autoconduck",
                        "messages": [{"role": "user", "content": S9_PROMPT}],
                    })
                elapsed = time.perf_counter() - started
                assert response.status_code == 200
                assert ANSWER in response.text
                canonical = []
                subagent_seen = 0
                for node in trace:
                    if node == "subagents":
                        subagent_seen += 1
                        canonical.append("recon_subagent_pool" if subagent_seen == 1 else "subagent_pool")
                    elif node != "idle":
                        canonical.append(node)
                expected = ["recon", "recon_subagent_pool", "planner", "subagent_pool", "compactor", "executor"]
                assert all(node in canonical for node in expected), canonical
                assert canonical.index("recon") < canonical.index("planner") < canonical.index("executor")
                calls = len(Mock.seen) - before
                print(f"S9 {label}: {elapsed:.2f}s")
                print(f"S9 {label} upstream call tally: {calls}")
                print(f"S9 {label} node trace: {canonical}")
                return elapsed
            finally:
                Mock.latency = 0

        ttft_0 = execute("TTFT_0s", 0)
        ttft_20 = execute("TTFT_20s", 20)
        print(f"S9 TTFT_0s: {ttft_0:.2f}s")
        print(f"S9 TTFT_20s: {ttft_20:.2f}s")
        print(f"S9 total upstream calls: {len(Mock.seen)}")
        return f"TTFT_0s={ttft_0:.2f}s TTFT_20s={ttft_20:.2f}s"

    def s12_tool_suppression():
        nonlocal cfg
        cfg = config(base); cm.get_config = lambda: cfg; server.get_config = lambda: cfg
        server.app = None; server._cached.clear(); trace.clear()
        dump = "\n".join(f"def generated_{i}(): return {i}" for i in range(40))
        messages = [{"role": "user", "content": S8_PROMPT if "S8_PROMPT" in globals() else plain_prompt}, {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}]}, {"role": "user", "content": dump}]
        from autoconduck.routing.dispatcher import route
        decision = route(messages, [], config=cfg)
        print("S12 routing decision:", {k: getattr(decision, k) for k in ("path", "confidence_band", "confidence", "complexity", "reason")})
        assert decision.path == "fast"
        with client() as c:
            before_slow = c.get("/stats").json().get("path_counts", {}).get("SLOW", 0)
            with c.stream("POST", "/v1/chat/completions", json={"model": "autoconduck", "messages": messages, "stream": True}) as response:
                text = _sse_content("".join(response.iter_text()))
            progress = [f"{label} ·" for label in ("recon", "planner", "subagents", "executor") if f"{label} ·" in text]
            path_counts = c.get("/stats").json().get("path_counts", {})
            print("S12 stream response status:", response.status_code)
            print("S12 stream response text:", text)
            print("S12 progress lines:", progress)
            print("S12 path counts:", path_counts)
            assert response.status_code == 200 and "Mock generic response." in text
            assert not progress, text
            assert path_counts.get("SLOW", 0) == before_slow
        return "pi user-role tool result stays fast"

    def s13_planner_recovery():
        nonlocal cfg
        cfg = config(base); cm.get_config = lambda: cfg; server.get_config = lambda: cfg
        server.app = None; server._cached.clear(); trace.clear(); Mock.planner_calls = 0
        with client() as c:
            response = c.post("/v1/chat/completions", json={"model": "autoconduck", "messages": [{"role": "user", "content": prompt}]})
            assert response.status_code == 200 and ANSWER in response.text
        canonical = []
        subagent_seen = 0
        for node in trace:
            if node == "subagents":
                subagent_seen += 1
                canonical.append("recon_subagent_pool" if subagent_seen == 1 else "subagent_pool")
            elif node != "idle":
                canonical.append(node)
        print("S13 planner recovery: plain JSON primary; response_format fallback is truncated")
        print("S13 planner calls:", Mock.planner_calls)
        print("S13 mock branches:", Mock.branches)
        print("S13 node trace:", canonical)
        print("S13 final response:", response.text)
        assert Mock.planner_calls == 1 and "TaskPlan-fallback" not in Mock.branches
        assert canonical == ["recon", "recon_subagent_pool", "planner", "subagent_pool", "subagent_pool", "compactor", "executor"]
        return "planner plain primary completed"

    def s15_json_repair_rescue():
        nonlocal cfg
        cfg = config(base); cm.get_config = lambda: cfg; server.get_config = lambda: cfg
        server.app = None; server._cached.clear(); trace.clear(); Mock.planner_calls = 0
        Mock.planner_truncation = True
        try:
            with client() as c:
                response = c.post("/v1/chat/completions", json={"model": "autoconduck", "messages": [{"role": "user", "content": prompt}]})
            assert response.status_code == 200 and ANSWER in response.text
            canonical = []
            subagent_seen = 0
            for node in trace:
                if node == "subagents":
                    subagent_seen += 1
                    canonical.append("recon_subagent_pool" if subagent_seen == 1 else "subagent_pool")
                elif node != "idle":
                    canonical.append(node)
            print("S15 repair rescue: unterminated planner JSON repaired on plain attempt")
            print("S15 node trace:", canonical)
            print("S15 final response:", response.text)
            assert canonical == ["recon", "recon_subagent_pool", "planner", "subagent_pool", "subagent_pool", "compactor", "executor"]
            return "unterminated string rescued; full pipeline completed"
        finally:
            Mock.planner_truncation = False

    def s14_planner_total_failure():
        nonlocal cfg
        cfg = config(base); cm.get_config = lambda: cfg; server.get_config = lambda: cfg
        server.app = None; server._cached.clear(); trace.clear(); Mock.planner_calls = 0
        Mock.planner_failure = True
        try:
            with client() as c:
                with c.stream("POST", "/v1/chat/completions", json={
                    "model": "autoconduck", "messages": [{"role": "user", "content": prompt}], "stream": True,
                }) as response:
                    text = _sse_content("".join(response.iter_text()))
            assert response.status_code == 200
            assert response.status_code == 200
            assert not any(f"{label} ·" in text for label in ("subagent", "executor")), text
            return "total planner failure degraded to FAST"
        finally:
            Mock.planner_failure = False

    def s16_cheaper_planner_retry():
        nonlocal cfg
        cfg = config(base); cm.get_config = lambda: cfg; server.get_config = lambda: cfg
        server.app = None; server._cached.clear(); trace.clear(); Mock.planner_calls = 0; Mock.planner_retry = True
        Mock.seen.clear()
        try:
            with client() as c:
                with c.stream("POST", "/v1/chat/completions", json={"model":"autoconduck", "messages":[{"role":"user","content":prompt}], "stream":True}) as response:
                    text = _sse_content("".join(response.iter_text()))
            planner_models = [b.get("model") for b in Mock.seen if any("coding-task planner" in str(m.get("content","")) for m in b.get("messages",[]))]
            assert len(planner_models) >= 2 and "mock-cheapest" in str(planner_models[-1])
            # Real deltas are ProgressFormatter (progress.py) glyph lines, e.g.
            # "⠋ recon · running · ..." / "⏳ N subagents" / "✓ Workflow completed.",
            # not the legacy bracketed "[recon]" strings.
            needles = ["recon ·", "planner ·", "subagents", "compactor ·", "executor ·", "Workflow completed.", ANSWER]
            assert all(n in text for n in needles), text
            print("S16 cheaper retry models:", planner_models)
            print("S16 verbose progress:", text)
            return "broken JSON retried with cheaper planner model"
        finally:
            Mock.planner_retry = False

    def s17_role_cards():
        nonlocal cfg
        cfg = config(base); cm.get_config = lambda: cfg; server.get_config = lambda: cfg
        server.app = None; server._cached.clear(); trace.clear(); Mock.planner_calls = 0; Mock.role_plan = True; Mock.seen.clear()
        try:
            with client() as c:
                with c.stream("POST", "/v1/chat/completions", json={"model":"autoconduck", "messages":[{"role":"user","content":prompt}], "stream":True}) as response:
                    text = _sse_content("".join(response.iter_text()))
            assert "subagent 1/2 (reviewer)" in text and "subagent 2/2 (worker)" in text, text
            sub = [b for b in Mock.seen if any("subagent" in str(m.get("content","")).lower() for m in b.get("messages",[]))]
            assert any("review subagent" in json.dumps(b).lower() and "single writer thread" not in json.dumps(b).lower() for b in sub)
            assert any("single writer thread" in json.dumps(b).lower() for b in sub)
            assert any("decision authority" in json.dumps(b).lower() for b in Mock.seen)
            assert any("scouting subagent" in json.dumps(b).lower() for b in Mock.seen)
            print("S17 role progress:", text)
            return "reviewer/worker cards and executor card verified"
        finally:
            Mock.role_plan = False

    run("S6 Anthropic", anthropic)
    run("S12 tool-turn suppression", s12_tool_suppression)
    run("S13 planner recovery", s13_planner_recovery)
    run("S15 repair rescue", s15_json_repair_rescue)
    run("S14 planner total failure", s14_planner_total_failure)
    run("S16 cheaper planner retry", s16_cheaper_planner_retry)
    run("S17 role cards", s17_role_cards)
    run("S9 measured slow path", s9_measure)
    upstream.shutdown(); LOG.unlink(missing_ok=True)
    print("\nScenario | Result | Notes\n---------|--------|------"); [print(" | ".join(x)) for x in results]
    return int(any(x[1] != "PASS" for x in results))

if __name__ == "__main__": raise SystemExit(main())
