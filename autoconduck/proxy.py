from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse, RedirectResponse
from pydantic import BaseModel, ConfigDict, Field

from .config import Config, ModelEntry, get_config
from . import gatekeeper as gatekeeper_mod
from . import evaluator as evaluator_mod
from . import pricing as pricing_mod
from . import state as state_mod
from . import telemetry as telemetry_mod
from . import cache as cache_mod
from .orchestrator import Orchestrator

# P2-8: tighten gatekeeper Decision.path to Literal (patch at import time)
try:
    _Lit = Literal["fast", "slow", "ambiguous"]  # type: ignore
    # patch annotation for fidelity; gatekeeper.Decision is a Pydantic model
    gatekeeper_mod.Decision.__annotations__["path"] = Literal["fast", "slow", "ambiguous"]  # type: ignore
    try:
        # pydantic v2 field annotation
        if hasattr(gatekeeper_mod.Decision, "model_fields") and "path" in gatekeeper_mod.Decision.model_fields:  # type: ignore
            gatekeeper_mod.Decision.model_fields["path"].annotation = Literal["fast", "slow", "ambiguous"]  # type: ignore
    except Exception:
        pass
except Exception:
    pass

# P2-10: attachment counting — patch gatekeeper._attachment_count to count multimodal parts
def _patched_attachment_count(request: Any) -> int:
    count = 0
    # explicit attachments field
    try:
        if hasattr(request, "attachments"):
            att = getattr(request, "attachments")
            if isinstance(att, (list, tuple)):
                count += len(att)
            elif att:
                count += 1
    except Exception:
        pass
    if isinstance(request, dict) and "attachments" in request:
        try:
            count += len(request.get("attachments") or [])
        except Exception:
            pass
    # count multimodal content parts (image_url, file, image) inside messages
    try:
        msgs = getattr(request, "messages", None)
        if msgs is None and isinstance(request, dict):
            msgs = request.get("messages", [])
        if msgs:
            for m in msgs:
                c = getattr(m, "content", None)
                if isinstance(m, dict):
                    c = m.get("content", c)
                if isinstance(c, list):
                    for part in c:
                        if isinstance(part, dict):
                            t = part.get("type", "")
                            if t in ("image_url", "file", "image", "input_image"):
                                count += 1
                            # also count if part has image_url/file key even without type
                            elif "image_url" in part or "file" in part:
                                count += 1
    except Exception:
        pass
    return count

try:
    gatekeeper_mod._attachment_count = _patched_attachment_count  # type: ignore
except Exception:
    pass

# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------

PSEUDO_MODELS = {"autoconduck", "autoconduck-budget", "autoconduck-expensive"}
DEFAULT_PORT = 11434
DISCONNECT_POLL_MS = 50

# max_in_flight semaphore (P1-6)
_in_flight_sem: asyncio.Semaphore | None = None


def _get_in_flight_sem() -> asyncio.Semaphore:
    global _in_flight_sem
    if _in_flight_sem is None:
        try:
            cfg = get_config()
            max_in_flight = int(getattr(cfg, "max_in_flight", 32))
        except Exception:
            max_in_flight = 32
        if max_in_flight <= 0:
            max_in_flight = 32
        _in_flight_sem = asyncio.Semaphore(max_in_flight)
    return _in_flight_sem


def _reset_in_flight_sem_for_tests(max_in_flight: int = 32) -> None:
    """Helper for tests to reset semaphore."""
    global _in_flight_sem
    _in_flight_sem = asyncio.Semaphore(max_in_flight)


def _extract_upstream_status(exc: BaseException) -> int | None:
    # direct attribute
    for attr in ("status_code", "status", "code"):
        v = getattr(exc, attr, None)
        if isinstance(v, int) and 100 <= v <= 599:
            return v
        # sometimes string numeric
        if isinstance(v, str) and v.isdigit():
            iv = int(v)
            if 100 <= iv <= 599:
                return iv
    # response object
    resp = getattr(exc, "response", None)
    if resp is not None:
        for attr in ("status_code", "status"):
            v = getattr(resp, attr, None)
            if isinstance(v, int) and 100 <= v <= 599:
                return v
    # cause chain
    cause = getattr(exc, "__cause__", None)
    if isinstance(cause, BaseException) and cause is not exc:
        c = _extract_upstream_status(cause)
        if c is not None:
            return c
    # context
    ctx = getattr(exc, "__context__", None)
    if isinstance(ctx, BaseException) and ctx is not exc:
        c = _extract_upstream_status(ctx)
        if c is not None:
            return c
    return None


def _extract_upstream_body(exc: BaseException) -> Any | None:
    # body attribute
    for attr in ("body", "message", "detail"):
        v = getattr(exc, attr, None)
        if isinstance(v, dict):
            return v
        if isinstance(v, (bytes, str)) and v:
            try:
                parsed = json.loads(v if isinstance(v, str) else v.decode("utf-8", errors="ignore"))
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
            # if string body looks like error, wrap
            if isinstance(v, (bytes, str)):
                s = v.decode("utf-8", errors="ignore") if isinstance(v, bytes) else v
                if s.strip().startswith("{"):
                    try:
                        return json.loads(s)
                    except Exception:
                        pass
    resp = getattr(exc, "response", None)
    if resp is not None:
        # httpx-like response
        try:
            if hasattr(resp, "json"):
                try:
                    j = resp.json()  # type: ignore
                    if isinstance(j, dict):
                        return j
                except Exception:
                    pass
            if hasattr(resp, "text"):
                t = getattr(resp, "text")
                if isinstance(t, str) and t.strip():
                    try:
                        pj = json.loads(t)
                        if isinstance(pj, dict):
                            return pj
                    except Exception:
                        return {"error": {"message": t[:500], "type": "upstream_error"}}
                elif isinstance(t, bytes):
                    try:
                        tt = t.decode("utf-8", errors="ignore")
                        pj = json.loads(tt)
                        if isinstance(pj, dict):
                            return pj
                    except Exception:
                        pass
            if hasattr(resp, "content"):
                c = getattr(resp, "content")
                if isinstance(c, (bytes, str)) and c:
                    s = c.decode("utf-8", errors="ignore") if isinstance(c, bytes) else c
                    try:
                        pj = json.loads(s)
                        if isinstance(pj, dict):
                            return pj
                    except Exception:
                        pass
        except Exception:
            pass
    # cause
    cause = getattr(exc, "__cause__", None)
    if isinstance(cause, BaseException) and cause is not exc:
        b = _extract_upstream_body(cause)
        if b is not None:
            return b
    return None


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"] = "user"
    content: Any = ""
    name: str | None = None
    tool_calls: Any | None = None
    tool_call_id: str | None = None
    model_config = ConfigDict(extra="allow")


class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    tools: Any | None = None
    tool_choice: Any | None = None
    model_config = ConfigDict(extra="allow")


@dataclass
class RequestContext:
    request_id: str
    chat: ChatRequest
    pseudo: str | None
    session_key: str
    turn_state: Any | None = None
    decision: Any | None = None
    T_i: float | None = None
    T_i_prime: float | None = None
    selected: ModelEntry | None = None
    orch: Any | None = None
    t0: float = 0.0
    steps_ms: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers: litellm forwarding
# ---------------------------------------------------------------------------

async def _forward_litellm_raw(chat: ChatRequest, model_id: str, stream: bool, extra_headers: dict | None = None):
    """
    Forward via litellm.acompletion without semaphore. Used for fallback.
    """
    msgs = []
    for m in chat.messages:
        d = m.model_dump(exclude_none=True)
        msgs.append(d)

    kwargs: dict[str, Any] = {
        "model": model_id,
        "messages": msgs,
        "stream": stream,
    }
    if chat.temperature is not None:
        kwargs["temperature"] = chat.temperature
    if chat.max_tokens is not None:
        kwargs["max_tokens"] = chat.max_tokens
    if chat.tools is not None:
        kwargs["tools"] = chat.tools
    if chat.tool_choice is not None:
        kwargs["tool_choice"] = chat.tool_choice
    extra = chat.model_extra or {}
    for k, v in extra.items():
        if k not in kwargs:
            kwargs[k] = v

    try:
        import litellm  # type: ignore

        if stream:
            resp = await litellm.acompletion(**kwargs)  # type: ignore
            return resp
        else:
            resp = await litellm.acompletion(**kwargs)  # type: ignore
            return resp
    except Exception as e:
        # Preserve upstream status if present — re-raise directly
        if _extract_upstream_status(e) is not None or hasattr(e, "status_code") or hasattr(e, "response"):
            raise
        # also check cause
        cause = getattr(e, "__cause__", None)
        if isinstance(cause, BaseException) and _extract_upstream_status(cause) is not None:
            raise
        raise RuntimeError(f"litellm forward failed: {e}") from e


async def _forward_litellm(chat: ChatRequest, model_id: str, stream: bool, extra_headers: dict | None = None):
    """
    Forward via litellm with max_in_flight semaphore. Timeout 5s on acquire.
    """
    sem = _get_in_flight_sem()
    try:
        await asyncio.wait_for(sem.acquire(), timeout=5.0)
    except asyncio.TimeoutError as e:
        raise TimeoutError("proxy overloaded: max_in_flight exceeded") from e
    try:
        return await _forward_litellm_raw(chat, model_id, stream, extra_headers=extra_headers)
    finally:
        try:
            sem.release()
        except Exception:
            pass


def _sse_from_litellm_chunk(chunk: Any) -> bytes | None:
    """
    Convert litellm chunk to SSE line. Returns None to skip.
    """
    try:
        if hasattr(chunk, "choices"):
            c0 = chunk.choices[0]  # type: ignore
            delta = getattr(c0, "delta", None)
            if delta is None and isinstance(c0, dict):
                delta = c0.get("delta")
            data = {}
            try:
                if hasattr(chunk, "model_dump"):
                    data = chunk.model_dump()  # type: ignore
                elif isinstance(chunk, dict):
                    data = chunk
                else:
                    data = json.loads(json.dumps(chunk, default=str))
            except Exception:
                data = {"choices": [{"delta": {"content": ""}}]}
            return f"data: {json.dumps(data)}\n\n".encode("utf-8")
        elif isinstance(chunk, dict):
            return f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
        else:
            return f"data: {json.dumps({'content': str(chunk)})}\n\n".encode("utf-8")
    except Exception:
        return None


async def _ambiguous_force_choice(last_text: str, cheap_model_id: str) -> str:
    """
    Single cheap forced-choice LLM call, 800ms timeout, returns FAST or SLOW.
    Wrapped with semaphore timeout 5s.
    """
    prompt = f'Decide FAST or SLOW for: "{last_text[:400]}"\nReply with one word: FAST or SLOW, then a one-line reason after " | ".'
    # semaphore guard for D4
    sem = _get_in_flight_sem()
    try:
        await asyncio.wait_for(sem.acquire(), timeout=5.0)
    except asyncio.TimeoutError as e:
        raise TimeoutError("proxy overloaded: max_in_flight exceeded") from e
    try:
        import litellm  # type: ignore

        resp = await asyncio.wait_for(
            litellm.acompletion(  # type: ignore
                model=cheap_model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0,
            ),
            timeout=0.8,
        )
        choice = resp.choices[0]  # type: ignore
        content = getattr(choice.message, "content", "") or ""
        text = str(content).strip().upper()
        if "SLOW" in text.split()[0]:
            return "SLOW"
        if "FAST" in text.split()[0]:
            return "FAST"
        if "SLOW" in text:
            return "SLOW"
        return "FAST"
    except TimeoutError:
        raise
    except asyncio.TimeoutError:
        # litellm timeout -> fallback FAST
        return "FAST"
    except Exception:
        return "FAST"
    finally:
        try:
            sem.release()
        except Exception:
            pass


def _get_cheapest_model_id(models: list[ModelEntry]) -> str | None:
    enabled = [m for m in models if m.enabled]
    if not enabled:
        return None
    enabled.sort(key=lambda m: (m.price_in + m.price_out))
    return enabled[0].id


# ---------------------------------------------------------------------------
# FastAPI app factory
# ---------------------------------------------------------------------------

def create_app(cfg: Config | None = None) -> FastAPI:
    if cfg is not None:
        try:
            import autoconduck.config as _cfgmod

            _cfgmod._config_singleton = cfg
        except Exception:
            pass
    else:
        cfg = get_config()
    # init semaphore from cfg
    global _in_flight_sem
    try:
        _in_flight_sem = asyncio.Semaphore(int(getattr(cfg, "max_in_flight", 32)) if getattr(cfg, "max_in_flight", 32) > 0 else 32)
    except Exception:
        _in_flight_sem = asyncio.Semaphore(32)
    app = FastAPI(title="AutoConduck Proxy", version="0.1.0")
    start_time = time.time()
    orchestrator = Orchestrator()

    try:
        state_mod.load_state()
    except Exception:
        pass

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok", "version": "0.1.0", "uptime": round(time.time() - start_time, 2), "port": cfg.port}

    @app.get("/")
    async def root():
        return RedirectResponse(url="/stats")

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {"id": "autoconduck", "object": "model", "created": 0, "owned_by": "autoconduck"},
                {"id": "autoconduck-budget", "object": "model", "created": 0, "owned_by": "autoconduck"},
                {"id": "autoconduck-expensive", "object": "model", "created": 0, "owned_by": "autoconduck"},
            ],
        }

    @app.get("/stats")
    async def stats():
        base = telemetry_mod.telemetry.stats()
        base["uptime_seconds"] = round(time.time() - start_time, 2)
        try:
            base["pricing_ema"] = state_mod.get_ema().to_dict()
            degraded = []
            for mid, w in state_mod._error_windows.items():  # type: ignore
                if state_mod.is_degraded(mid):
                    degraded.append(f"{mid}: {w.error_rate():.2f} error rate (5m)")
            base["degraded_models"] = degraded
        except Exception:
            pass
        return base

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        t0 = time.perf_counter()
        request_id = str(uuid.uuid4())[:8]
        steps_ms: dict[str, float] = {}
        body_bytes: bytes | None = None
        try:
            raw = await request.body()
            body_bytes = raw
            data = json.loads(raw.decode("utf-8") if raw else "{}")
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": f"invalid JSON: {e}", "type": "invalid_request_error", "code": 400}},
            )

        try:
            chat = ChatRequest.model_validate(data)
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": f"validation error: {e}", "type": "invalid_request_error", "code": 400}},
            )

        if not chat.messages:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "messages is required", "type": "invalid_request_error", "code": 400}},
            )

        cfg_local = get_config()

        # Thread RequestContext (P2-9) — instantiate early
        pseudo = chat.model if chat.model in PSEUDO_MODELS else None
        headers_dict = dict(request.headers)
        session_key = state_mod.session_key_from_request(chat, headers_dict)
        turn_state = state_mod.get_session_store().get(session_key)
        ctx = RequestContext(
            request_id=request_id,
            chat=chat,
            pseudo=pseudo,
            session_key=session_key,
            turn_state=turn_state,
            t0=t0,
            steps_ms=steps_ms,
        )

        if pseudo is None:
            # Passthrough with semaphore + upstream pass-through
            t_passthrough_start = time.perf_counter()
            try:
                if chat.stream:
                    try:
                        upstream = await _forward_litellm(chat, chat.model, True)
                    except TimeoutError as e:
                        return JSONResponse(
                            status_code=503,
                            content={"error": {"message": "proxy overloaded, retry later", "type": "proxy_overload", "code": 503}},
                            headers={"Retry-After": "2"},
                        )
                    async def gen():
                        try:
                            async for chunk in upstream:  # type: ignore
                                if await request.is_disconnected():
                                    try:
                                        await upstream.aclose()  # type: ignore
                                    except Exception:
                                        pass
                                    break
                                line = _sse_from_litellm_chunk(chunk)
                                if line:
                                    yield line
                            yield b"data: [DONE]\n\n"
                        except asyncio.CancelledError:
                            return
                        except Exception as e:
                            yield f"data: {json.dumps({'error': str(e)})}\n\n".encode()
                    resp = StreamingResponse(gen(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no", "x-autoconduck-model": chat.model})
                    evt = telemetry_mod.RoutingEvent(
                        ts=time.time(),
                        request_id=request_id,
                        pseudo_model=None,
                        real_model=chat.model,
                        path="passthrough",
                        gate_reason="passthrough",
                        latency_overhead_ms=(time.perf_counter() - t0) * 1000,
                    )
                    telemetry_mod.telemetry.push(evt)
                    return resp
                else:
                    try:
                        resp_obj = await _forward_litellm(chat, chat.model, False)
                    except TimeoutError as e:
                        return JSONResponse(
                            status_code=503,
                            content={"error": {"message": "proxy overloaded, retry later", "type": "proxy_overload", "code": 503}},
                            headers={"Retry-After": "2"},
                        )
                    try:
                        body = resp_obj.model_dump() if hasattr(resp_obj, "model_dump") else dict(resp_obj)  # type: ignore
                    except Exception:
                        body = {"choices": [{"message": {"content": str(resp_obj)}}]}
                    evt = telemetry_mod.RoutingEvent(
                        ts=time.time(),
                        request_id=request_id,
                        pseudo_model=None,
                        real_model=chat.model,
                        path="passthrough",
                        latency_overhead_ms=(time.perf_counter() - t0) * 1000,
                    )
                    telemetry_mod.telemetry.push(evt)
                    return JSONResponse(content=body, headers={"x-autoconduck-model": chat.model})
            except TimeoutError:
                return JSONResponse(
                    status_code=503,
                    content={"error": {"message": "proxy overloaded, retry later", "type": "proxy_overload", "code": 503}},
                    headers={"Retry-After": "2"},
                )
            except Exception as e:
                # upstream pass-through (P1-5)
                sc = _extract_upstream_status(e)
                if sc is not None:
                    upstream_body = _extract_upstream_body(e)
                    if upstream_body is None:
                        # try to parse message
                        try:
                            upstream_body = {"error": {"message": str(e), "type": "upstream_error", "code": sc}}
                        except Exception:
                            upstream_body = {"error": {"message": "upstream error", "type": "upstream_error", "code": sc}}
                    # ensure code matches
                    if isinstance(upstream_body, dict) and "error" in upstream_body and isinstance(upstream_body["error"], dict):
                        upstream_body["error"].setdefault("code", sc)
                    return JSONResponse(status_code=sc, content=upstream_body)
                state_mod.record_error(chat.model)
                return JSONResponse(status_code=502, content={"error": {"message": str(e), "type": "proxy_error", "code": 502}})

        # D2 cache lookup
        if cfg_local.cache_enabled:
            try:
                last_msg = chat.messages[-1]
                key = cache_mod.make_key(pseudo, last_msg)
                cached = cache_mod.get(key)
                if cached is not None:
                    async def gen_cached():
                        yield cached
                    evt = telemetry_mod.RoutingEvent(
                        ts=time.time(),
                        request_id=request_id,
                        pseudo_model=pseudo,
                        real_model="cache",
                        path="cache_hit",
                        cache_hit=True,
                        latency_overhead_ms=(time.perf_counter() - t0) * 1000,
                    )
                    telemetry_mod.telemetry.push(evt)
                    if cached.lstrip().startswith(b"data:"):
                        return StreamingResponse(gen_cached(), media_type="text/event-stream")
                    else:
                        return Response(content=cached, media_type="application/json")
            except Exception:
                pass

        # session key + turn state already in ctx
        # ctx.turn_state already set

        # D3 classify
        t_class_start = time.perf_counter()
        decision = gatekeeper_mod.classify(chat, turn_state)
        steps_ms["classify"] = (time.perf_counter() - t_class_start) * 1000
        ctx.decision = decision

        # D4 ambiguous — with semaphore timeout graceful degrade
        path = decision.path
        T_i: float | None = decision.T_i
        ctx.T_i = T_i
        ambiguous_resolved: str | None = None

        if path == "ambiguous":
            cheap_id = _get_cheapest_model_id(cfg_local.models)
            last_text = ""
            try:
                last_msg = chat.messages[-1]
                c = getattr(last_msg, "content", "")
                if isinstance(c, list):
                    c = " ".join(str(x.get("text", x) if isinstance(x, dict) else str(x)) for x in c)
                last_text = str(c)
            except Exception:
                last_text = ""
            if cheap_id:
                t_amb = time.perf_counter()
                try:
                    choice = await _ambiguous_force_choice(last_text, cheap_id)
                except TimeoutError:
                    # semaphore overloaded — degrade to fast without LLM call
                    choice = "FAST"
                    steps_ms["ambiguous_timeout"] = (time.perf_counter() - t_amb) * 1000
                    # also mark degraded
                    path = "fast"
                    ambiguous_resolved = "ambiguous_timeout_fallback"
                else:
                    steps_ms["ambiguous_llm"] = (time.perf_counter() - t_amb) * 1000
                    if choice == "SLOW":
                        path = "slow"
                        ambiguous_resolved = "ambiguous_resolved_slow"
                    else:
                        path = "fast"
                        ambiguous_resolved = "ambiguous_resolved_fast"
                if ambiguous_resolved is None:
                    # if not timeout path, already set above
                    pass
            else:
                path = "fast"
                ambiguous_resolved = "ambiguous_resolved_fast"

        # D5 ensure T_i
        if T_i is None:
            t_score = time.perf_counter()
            try:
                last_msg = chat.messages[-1]
                T_i = float(evaluator_mod.score(last_msg, turn_state))
            except Exception:
                T_i = 0.5
            steps_ms["score"] = (time.perf_counter() - t_score) * 1000
        T_i = max(0.0, min(1.0, float(T_i)))  # type: ignore
        ctx.T_i = T_i

        # Orchestrator if slow — D6 via pricing.select (P0-1) + O6 error accounting (P0-2)
        compacted: str | None = None
        orch_result = None
        degraded_to_fast = False
        worker_ok = None
        worker_fail = None
        plan_model_id: str | None = None
        worker_model_id: str | None = None

        if path == "slow":
            # D6: select plan/worker via pricing.select(0.40/0.55)
            try:
                # estimate tokens for pricing selection
                t_in_est, t_out_est = pricing_mod.estimate_tokens(chat.messages, cfg_local, intent=None)
                # select returns ModelEntry
                try:
                    plan_entry = pricing_mod.select(0.40, cfg_local.models, t_in_est, t_out_est)
                    plan_model_id = plan_entry.id
                except Exception:
                    # fallback if select fails (e.g. no models) — will degrade
                    raise
                try:
                    worker_entry = pricing_mod.select(0.55, cfg_local.models, t_in_est, t_out_est)
                    worker_model_id = worker_entry.id
                except Exception:
                    worker_model_id = plan_model_id

                # stash in ctx
                ctx.orch = {"plan_model_id": plan_model_id, "worker_model_id": worker_model_id}

                t_orch = time.perf_counter()
                orch_result = await orchestrator.plan_and_execute(
                    chat, plan_model_id=plan_model_id, worker_model_id=worker_model_id
                )
                steps_ms["orchestrator"] = (time.perf_counter() - t_orch) * 1000
                worker_ok = orch_result.worker_ok
                worker_fail = orch_result.worker_fail
                ctx.orch = orch_result
                # O6: worker/plan error accounting
                try:
                    # plan failure: plan is None and degraded_to_fast
                    if orch_result.degraded_to_fast and orch_result.plan is None and orch_result.plan_model_id:
                        pricing_mod.record_error(orch_result.plan_model_id)
                    elif orch_result.degraded_to_fast and orch_result.plan is None and plan_model_id:
                        pricing_mod.record_error(plan_model_id)
                except Exception:
                    pass
                try:
                    # worker failures: one record_error per failed worker's actual model id
                    if orch_result.worker_fail and orch_result.worker_fail > 0:
                        n_fail = int(orch_result.worker_fail)
                        wids = list(getattr(orch_result, "worker_model_ids", None) or [])
                        if wids:
                            # Prefer per-worker ids from OrchestratorResult (failed workers)
                            for wid in wids[:n_fail]:
                                try:
                                    pricing_mod.record_error(wid)
                                except Exception:
                                    pass
                            # Pad if list shorter than fail count
                            if len(wids) < n_fail:
                                pad = worker_model_id or wids[0] or plan_model_id
                                if pad:
                                    for _ in range(n_fail - len(wids)):
                                        try:
                                            pricing_mod.record_error(pad)
                                        except Exception:
                                            pass
                        else:
                            # fallback shared worker_model_id only when per-worker ids unavailable
                            wid = worker_model_id or plan_model_id
                            if wid:
                                for _ in range(n_fail):
                                    try:
                                        pricing_mod.record_error(wid)
                                    except Exception:
                                        pass
                except Exception:
                    pass

                if orch_result.degraded_to_fast:
                    degraded_to_fast = True
                    path = "fast"
                else:
                    compacted = orch_result.compacted_context
            except Exception as e:
                # If pricing.select failed due to empty models, let D7 handle empty list 503
                # For other orchestrator errors, degrade to fast
                # Check if it's due to no enabled models — not degraded flag yet, let outer handle
                # Detect ValueError no enabled models
                if "no enabled models" in str(e).lower() or "no models" in str(e).lower():
                    # will be handled in D7 select — mark degraded
                    degraded_to_fast = True
                else:
                    degraded_to_fast = True
                steps_ms["orchestrator_error"] = 0
                # If we have plan_model_id but no orch_result, still record plan error if applicable
                # Not enough info, skip

        # D7 transform + estimate + select
        t_sel = time.perf_counter()
        T_prime = pricing_mod.transform(float(T_i), pseudo)
        ctx.T_i_prime = T_prime
        text_for_intent = ""
        try:
            last = chat.messages[-1]
            c = getattr(last, "content", "")
            if isinstance(c, list):
                c = " ".join(str(x.get("text", x) if isinstance(x, dict) else str(x)) for x in c)
            text_for_intent = str(c)
        except Exception:
            text_for_intent = ""
        t_in, t_out = pricing_mod.estimate_tokens(chat.messages, cfg_local, intent=None)
        try:
            selected = pricing_mod.select(T_prime, cfg_local.models, t_in, t_out)
            ctx.selected = selected
        except Exception:
            enabled = [m for m in cfg_local.models if m.enabled]
            if enabled:
                enabled.sort(key=lambda m: m.price_in + m.price_out)
                selected = enabled[0]
                ctx.selected = selected
            else:
                return JSONResponse(status_code=503, content={"error": {"message": "no models configured", "type": "config_error", "code": 503}})
        steps_ms["select"] = (time.perf_counter() - t_sel) * 1000

        # mutate payload
        original_model = chat.model
        chat.model = selected.id
        if compacted and not degraded_to_fast:
            sys_msg = Message(role="system", content=f"[AutoConduck context]\n{compacted}")
            idx = 0
            for i, m in enumerate(chat.messages):
                if m.role == "system":
                    idx = i + 1
                else:
                    break
            chat.messages.insert(idx, sys_msg)

        # D8 forward
        overhead_ms = (time.perf_counter() - t0) * 1000
        cost_est = (selected.price_in / 1000.0) * t_in + (selected.price_out / 1000.0) * t_out

        tele_path = path
        if ambiguous_resolved:
            tele_path = ambiguous_resolved

        try:
            if chat.stream:
                # try to forward with semaphore; on TimeoutError fallback to cheapest directly (P1-6)
                try:
                    upstream = await _forward_litellm(chat, selected.id, True)
                except TimeoutError:
                    # Pseudo-model fallback: cheapest model directly (skip orchestration already done, but we still try cheapest)
                    # If already selected is cheapest, just return 503 overload
                    cheapest_id = _get_cheapest_model_id(cfg_local.models)
                    if cheapest_id and cheapest_id != selected.id:
                        # try raw forward without semaphore (bypass overload) to still serve
                        try:
                            upstream = await _forward_litellm_raw(chat, cheapest_id, True)
                            selected = [m for m in cfg_local.models if m.id == cheapest_id][0] if any(m.id == cheapest_id for m in cfg_local.models) else selected
                        except Exception as e2:
                            sc = _extract_upstream_status(e2)
                            if sc is not None:
                                ub = _extract_upstream_body(e2) or {"error": {"message": str(e2), "type": "upstream_error", "code": sc}}
                                return JSONResponse(status_code=sc, content=ub)
                            return JSONResponse(status_code=503, content={"error": {"message": "proxy overloaded, retry later", "type": "proxy_overload", "code": 503}}, headers={"Retry-After": "2"})
                    else:
                        return JSONResponse(status_code=503, content={"error": {"message": "proxy overloaded, retry later", "type": "proxy_overload", "code": 503}}, headers={"Retry-After": "2"})
                collected_chunks: list[bytes] = []

                async def gen_stream():
                    actual_in = t_in
                    actual_out = 0
                    chunk_count = 0
                    try:
                        async for chunk in upstream:  # type: ignore
                            if await request.is_disconnected():
                                try:
                                    if hasattr(upstream, "aclose"):
                                        await upstream.aclose()  # type: ignore
                                except Exception:
                                    pass
                                raise asyncio.CancelledError()
                            line = _sse_from_litellm_chunk(chunk)
                            if line:
                                collected_chunks.append(line)
                                try:
                                    d = json.loads(line[len(b"data: ") :].strip() or b"{}")
                                    if isinstance(d, dict) and "usage" in d:
                                        u = d["usage"]
                                        actual_in = int(u.get("prompt_tokens", actual_in))
                                        actual_out = int(u.get("completion_tokens", actual_out))
                                except Exception:
                                    pass
                                yield line
                            chunk_count += 1
                        yield b"data: [DONE]\n\n"
                        try:
                            if actual_out == 0:
                                actual_out = max(1, len(b"".join(collected_chunks)) // 4)
                            pricing_mod.record_usage(selected.id, actual_in, actual_out, intent=pricing_mod._detect_intent(text_for_intent))
                            state_mod.update_turn_state(session_key, T_prime, selected.tier)
                        except Exception:
                            pass
                    except asyncio.CancelledError:
                        # Disconnect ≠ error (P0-3): do NOT call pricing.record_error
                        evt2 = telemetry_mod.RoutingEvent(
                            ts=time.time(),
                            request_id=request_id,
                            pseudo_model=pseudo,
                            real_model=selected.id,
                            path=tele_path,
                            gate_reason=decision.reason,
                            T_i=float(T_i),
                            T_i_prime=float(T_prime),
                            degraded_to_fast=degraded_to_fast,
                            cost_est=cost_est,
                            latency_overhead_ms=overhead_ms,
                            cancelled=True,
                            worker_ok=worker_ok,
                            worker_fail=worker_fail,
                            steps_ms=steps_ms,
                        )
                        telemetry_mod.telemetry.push(evt2)
                        return
                    except Exception as e:
                        yield f"data: {json.dumps({'error': str(e)})}\n\n".encode()

                async def gen_with_telemetry():
                    try:
                        async for part in gen_stream():
                            yield part
                        evt = telemetry_mod.RoutingEvent(
                            ts=time.time(),
                            request_id=request_id,
                            pseudo_model=pseudo,
                            real_model=selected.id,
                            path=tele_path,
                            gate_reason=decision.reason,
                            T_i=float(T_i),
                            T_i_prime=float(T_prime),
                            degraded_to_fast=degraded_to_fast,
                            cost_est=cost_est,
                            latency_overhead_ms=overhead_ms,
                            worker_ok=worker_ok,
                            worker_fail=worker_fail,
                            steps_ms=steps_ms,
                        )
                        telemetry_mod.telemetry.push(evt)
                        if cfg_local.cache_enabled and collected_chunks:
                            try:
                                key = cache_mod.make_key(pseudo, chat.messages[-1] if chat.messages else "")
                                body = b"".join(collected_chunks) + b"data: [DONE]\n\n"
                                cache_mod.put(key, body)
                            except Exception:
                                pass
                    except Exception:
                        pass

                return StreamingResponse(gen_with_telemetry(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache", "x-autoconduck-model": selected.id})

            else:
                # non-streaming
                try:
                    resp_obj = await _forward_litellm(chat, selected.id, False)
                except TimeoutError:
                    cheapest_id = _get_cheapest_model_id(cfg_local.models)
                    if cheapest_id and cheapest_id != selected.id:
                        try:
                            resp_obj = await _forward_litellm_raw(chat, cheapest_id, False)
                            # update selected for telemetry
                            try:
                                selected = next(m for m in cfg_local.models if m.id == cheapest_id)
                                ctx.selected = selected
                            except Exception:
                                pass
                        except Exception as e2:
                            sc = _extract_upstream_status(e2)
                            if sc is not None:
                                ub = _extract_upstream_body(e2) or {"error": {"message": str(e2), "type": "upstream_error", "code": sc}}
                                return JSONResponse(status_code=sc, content=ub)
                            return JSONResponse(status_code=503, content={"error": {"message": "proxy overloaded, retry later", "type": "proxy_overload", "code": 503}}, headers={"Retry-After": "2"})
                    else:
                        return JSONResponse(status_code=503, content={"error": {"message": "proxy overloaded, retry later", "type": "proxy_overload", "code": 503}}, headers={"Retry-After": "2"})
                if await request.is_disconnected():
                    evt = telemetry_mod.RoutingEvent(
                        ts=time.time(),
                        request_id=request_id,
                        pseudo_model=pseudo,
                        real_model=selected.id,
                        path=tele_path,
                        gate_reason=decision.reason,
                        T_i=float(T_i),
                        T_i_prime=float(T_prime),
                        degraded_to_fast=degraded_to_fast,
                        cost_est=cost_est,
                        latency_overhead_ms=overhead_ms,
                        cancelled=True,
                        worker_ok=worker_ok,
                        worker_fail=worker_fail,
                        steps_ms=steps_ms,
                    )
                    telemetry_mod.telemetry.push(evt)
                    return Response(status_code=499, content=b"client disconnected")

                try:
                    if hasattr(resp_obj, "model_dump"):
                        body_dict = resp_obj.model_dump()  # type: ignore
                    elif isinstance(resp_obj, dict):
                        body_dict = resp_obj
                    else:
                        body_dict = {"choices": [{"message": {"content": str(resp_obj)}}]}
                except Exception:
                    body_dict = {"choices": [{"message": {"content": str(resp_obj)}}]}

                try:
                    usage = body_dict.get("usage") if isinstance(body_dict, dict) else None
                    if usage:
                        ai = int(usage.get("prompt_tokens", t_in))
                        ao = int(usage.get("completion_tokens", t_out))
                        pricing_mod.record_usage(selected.id, ai, ao, intent=pricing_mod._detect_intent(text_for_intent))
                    else:
                        pricing_mod.record_usage(selected.id, t_in, t_out, intent=pricing_mod._detect_intent(text_for_intent))
                    state_mod.update_turn_state(session_key, T_prime, selected.tier)
                except Exception:
                    pass

                evt = telemetry_mod.RoutingEvent(
                    ts=time.time(),
                    request_id=request_id,
                    pseudo_model=pseudo,
                    real_model=selected.id,
                    path=tele_path,
                    gate_reason=decision.reason,
                    T_i=float(T_i),
                    T_i_prime=float(T_prime),
                    degraded_to_fast=degraded_to_fast,
                    cost_est=cost_est,
                    latency_overhead_ms=overhead_ms,
                    worker_ok=worker_ok,
                    worker_fail=worker_fail,
                    steps_ms=steps_ms,
                )
                telemetry_mod.telemetry.push(evt)

                if cfg_local.cache_enabled:
                    try:
                        key = cache_mod.make_key(pseudo, chat.messages[-1] if chat.messages else "")
                        cache_mod.put(key, json.dumps(body_dict).encode("utf-8"))
                    except Exception:
                        pass

                return JSONResponse(content=body_dict, headers={"x-autoconduck-model": selected.id})
        except asyncio.CancelledError:
            # Disconnect ≠ error: no record_error
            evt = telemetry_mod.RoutingEvent(
                ts=time.time(),
                request_id=request_id,
                pseudo_model=pseudo,
                real_model=selected.id,
                path=tele_path,
                gate_reason=decision.reason,
                T_i=float(T_i),
                T_i_prime=float(T_prime),
                degraded_to_fast=degraded_to_fast,
                cost_est=cost_est,
                latency_overhead_ms=overhead_ms,
                cancelled=True,
                steps_ms=steps_ms,
            )
            telemetry_mod.telemetry.push(evt)
            return Response(status_code=499, content=b"cancelled")
        except TimeoutError as e:
            # already handled inner, but fallback for any remaining timeout
            return JSONResponse(status_code=503, content={"error": {"message": "proxy overloaded, retry later", "type": "proxy_overload", "code": 503}}, headers={"Retry-After": "2"})
        except Exception as e:
            # Upstream pass-through check (P1-5)
            sc = _extract_upstream_status(e)
            if sc is not None:
                upstream_body = _extract_upstream_body(e)
                if upstream_body is None:
                    upstream_body = {"error": {"message": str(e), "type": "upstream_error", "code": sc}}
                if isinstance(upstream_body, dict) and "error" in upstream_body and isinstance(upstream_body["error"], dict):
                    upstream_body["error"].setdefault("code", sc)
                # Still record error for 5xx? but not for 4xx — for now record for 5xx
                if sc >= 500:
                    try:
                        pricing_mod.record_error(selected.id)
                    except Exception:
                        pass
                evt = telemetry_mod.RoutingEvent(
                    ts=time.time(),
                    request_id=request_id,
                    pseudo_model=pseudo,
                    real_model=selected.id,
                    path=tele_path,
                    gate_reason=decision.reason,
                    T_i=float(T_i),
                    T_i_prime=float(T_prime),
                    degraded_to_fast=degraded_to_fast,
                    cost_est=cost_est,
                    latency_overhead_ms=overhead_ms,
                    error=str(e)[:500],
                    worker_ok=worker_ok,
                    worker_fail=worker_fail,
                    steps_ms=steps_ms,
                )
                telemetry_mod.telemetry.push(evt)
                return JSONResponse(status_code=sc, content=upstream_body)
            try:
                pricing_mod.record_error(selected.id)
            except Exception:
                pass
            evt = telemetry_mod.RoutingEvent(
                ts=time.time(),
                request_id=request_id,
                pseudo_model=pseudo,
                real_model=selected.id,
                path=tele_path,
                gate_reason=decision.reason,
                T_i=float(T_i),
                T_i_prime=float(T_prime),
                degraded_to_fast=degraded_to_fast,
                cost_est=cost_est,
                latency_overhead_ms=overhead_ms,
                error=str(e)[:500],
                worker_ok=worker_ok,
                worker_fail=worker_fail,
                steps_ms=steps_ms,
            )
            telemetry_mod.telemetry.push(evt)
            return JSONResponse(status_code=502, content={"error": {"message": str(e), "type": "proxy_error", "code": 502}})

    return app


# Default app for uvicorn import string
app = create_app()
