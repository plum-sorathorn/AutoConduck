from dataclasses import dataclass
from typing import Literal
import os
from urllib.parse import urlsplit
from . import semantic_router, evaluator, pricing


def _user_messages(messages: list) -> list:
    return [
        message
        for message in messages
        if not isinstance(message, dict) or message.get("role", "user") == "user"
    ]


@dataclass(frozen=True)
class RoutingDecision:
    path: Literal["fast", "slow"]
    confidence_band: Literal["fast", "slow", "ambiguous"]
    confidence: float
    complexity: float
    reason: str
    model: str | None = None


def _default_tiebreaker(message, pseudo_model, config):
    try:
        import litellm
        from ..config import (
            get_config,
            normalize_api_base,
            qualify_model,
            resolve_api_key,
        )

        cfg = config or get_config()
        content = getattr(message, "content", str(message))
        model = pricing.cheapest_enabled(cfg)
        if not model:
            return None
        entry = next(
            (
                e
                for source in (
                    getattr(cfg, "model_list", []) or [],
                    getattr(cfg, "custom_models", []) or [],
                )
                for e in source
                if isinstance(e, dict)
                and str(
                    e.get("id") or e.get("model_name") or e.get("model") or ""
                ).removeprefix("openai/")
                == str(model).removeprefix("openai/")
            ),
            {},
        )
        source = (
            entry.get("litellm_params")
            if isinstance(entry.get("litellm_params"), dict)
            else entry
        )
        base = source.get("base_url") or source.get("api_base")
        if base:
            base = normalize_api_base(base)
            parsed = urlsplit(base)
            try:
                local_port = int(
                    os.environ.get("AUTOCONDUCK_PORT", getattr(cfg, "port", 11434))
                )
            except ValueError:
                local_port = getattr(cfg, "port", 11434)
            if (
                parsed.hostname in {"127.0.0.1", "localhost", "::1"}
                and (parsed.port or 80) == local_port
            ):
                # The local gateway would recurse into this tiebreaker call.
                return None
        params = {"model": qualify_model(model)}
        if base:
            params["api_base"] = base
        api_key = resolve_api_key(source)
        if api_key:
            params["api_key"] = api_key
        short_content = str(content or "")[:500]
        result = litellm.completion(
            messages=[
                {
                    "role": "user",
                    "content": f"Reply with FAST or SLOW, then a space, then a complexity digit 1-9 (1=trivial, 9=very complex). Example: 'FAST 3'. Classify: {short_content}",
                }
            ],
            max_tokens=3,
            timeout=1.5,
            **params,
        )
        answer = result.choices[0].message.content.strip().upper()
        match = __import__("re").match(
            r"^(FAST|SLOW)(?:\s+([1-9]))?\s*$", answer
        )
        return answer if match else None
    except Exception:
        return None


def route(
    messages: list,
    history,
    pseudo_model: str = "autoconduck",
    tiebreaker=None,
    config=None,
) -> RoutingDecision:
    if config is None:
        from ..config import get_config

        config = get_config()

    # Step 1: Input text extraction & sanitization
    user_msgs = _user_messages(messages)
    last = user_msgs[-1] if user_msgs else ""
    text = (
        last.get("content", "")
        if isinstance(last, dict)
        else getattr(last, "content", str(last))
    )
    raw_text = evaluator.clean_routing_text(text)

    # Step 2: Semantic route matching
    route_match = semantic_router.route(raw_text)

    # Step 3: Complexity scoring and factor evaluation
    score = evaluator.score(messages, history, route_match, pseudo_model, config)
    path = getattr(score, "path", "fast")
    confidence_band = getattr(score, "confidence_band", "fast")
    confidence = getattr(score, "confidence", 0.0)
    complexity = getattr(score, "complexity", 0.0)
    reason = getattr(score, "reason", "")

    # Step 4: Ambiguous tiebreaker resolution
    if confidence_band == "ambiguous":
        enabled = [
            entry
            for entry in (getattr(config, "model_list", []) or [])
            if entry.get("enabled", True)
        ]
        if len(enabled) <= 1:
            path = "slow" if route_match.route == "slow_path" else "fast"
            reason = f"single-model, router-resolved: {path}"
        else:
            selection = getattr(config, "selection", config)
            tiebreaker_floor = float(getattr(selection, "tiebreaker_min_complexity", 0.45))
            if pseudo_model.endswith("budget"):
                tiebreaker_floor = float(
                    getattr(selection, "budget_tiebreaker_min_complexity", 0.65)
                )
            use_tiebreaker = tiebreaker is not None or (
                bool(getattr(selection, "tiebreaker_enabled", True))
                and complexity >= tiebreaker_floor
            )
            if not use_tiebreaker:
                path = "fast"
                reason = "tiebreaker: fast (below-floor)"
            else:
                try:
                    tb_fn = tiebreaker or _default_tiebreaker
                    last_msg = messages[-1] if messages else ""
                    answer = str(tb_fn(last_msg, pseudo_model, config)).upper()
                except Exception:
                    answer = "NONE"

                import re

                m = re.match(r"^(FAST|SLOW)(?:\s+([1-9]))?\s*$", answer)
                if answer == "NONE" or not m:
                    slow_threshold = float(getattr(selection, "slow_threshold", 0.75))
                    tiebreaker_model = pricing.cheapest_enabled(config)
                    degraded = bool(
                        tiebreaker_model
                        and pricing.is_degraded(
                            tiebreaker_model,
                            getattr(config, "degraded_window_s", 300),
                            getattr(config, "degraded_error_rate", 0.2),
                        )
                    )
                    if degraded:
                        path = "fast"
                        reason_suffix = "unavailable: degraded-provider"
                    else:
                        path = "slow" if complexity >= slow_threshold else "fast"
                        reason_suffix = "unavailable: complexity-fallback"
                    reason = f"tiebreaker_{reason_suffix}"
                else:
                    digit = int(m.group(2)) if m.group(2) else None
                    if digit is not None:
                        complexity = 0.5 * complexity + 0.5 * (digit / 9)
                    path = "slow" if m.group(1) == "SLOW" else "fast"
                    reason = f"tiebreaker: {path}"

    # Step 5: Dynamic closest-cost model selection
    model = None
    if path == "fast":
        from ..config import resolve_orchestrator_model

        selection = getattr(config, "selection", config)
        max_fast_cost = getattr(selection, "fast_path_max_scaled_cost", 0.50)
        try:
            max_fast_cost = float(max_fast_cost)
        except (TypeError, ValueError):
            max_fast_cost = 0.50

        per_turn_enabled = bool(getattr(selection, "enable_per_turn_task_routing", True))
        turn_task = evaluator.detect_turn_task(messages) if per_turn_enabled else None

        turn_complexity = complexity
        band = None
        if turn_task == "recon":
            recon_max = float(getattr(selection, "recon_max_complexity", 0.20))
            turn_complexity = min(turn_complexity, recon_max)
            recon_band = getattr(selection, "recon_task_band", [0.05, 0.35])
            if isinstance(recon_band, (list, tuple)) and len(recon_band) == 2:
                max_fast_cost = min(max_fast_cost, float(recon_band[1]))
        elif turn_task == "edit":
            edit_min = float(getattr(selection, "edit_min_complexity", 0.45))
            turn_complexity = max(turn_complexity, edit_min)
            edit_band = getattr(selection, "edit_task_band", [0.30, 0.65])
            if isinstance(edit_band, (list, tuple)) and len(edit_band) == 2:
                max_fast_cost = max(max_fast_cost, float(edit_band[1]))
        elif turn_task == "verify":
            verify_band = getattr(selection, "verify_task_band", [0.15, 0.50])
            if isinstance(verify_band, (list, tuple)) and len(verify_band) == 2:
                turn_complexity = max(float(verify_band[0]), min(float(verify_band[1]), turn_complexity))
                max_fast_cost = min(max_fast_cost, float(verify_band[1]))
        elif turn_task == "bash":
            bash_band = getattr(selection, "bash_task_band", [0.20, 0.55])
            if isinstance(bash_band, (list, tuple)) and len(bash_band) == 2:
                turn_complexity = max(float(bash_band[0]), min(float(bash_band[1]), turn_complexity))
                max_fast_cost = min(max_fast_cost, float(bash_band[1]))

        model = pricing.select_closest(
            pricing.pool_ids(config),
            turn_complexity,
            config,
            pseudo_model=pseudo_model,
            band=band,
            max_scaled_cost=max_fast_cost,
        ) or resolve_orchestrator_model(config)
        if model:
            from ..stats import record_selection

            record_selection(
                complexity,
                pricing.target_scaled_cost(complexity, pseudo_model, config),
                model,
                config,
            )

    return RoutingDecision(
        path=path,
        confidence_band=confidence_band,
        confidence=confidence,
        complexity=complexity,
        reason=reason,
        model=model,
    )

