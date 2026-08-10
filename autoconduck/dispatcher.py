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
        from .config import (
            get_config,
            normalize_api_base,
            qualify_model,
            resolve_api_key,
        )

        cfg = config or get_config()
        content = getattr(message, "content", str(message))
        model = pricing.cheapest_enabled(cfg)
        if not model:
            return "fast"
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
                return "fast"
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
        return answer if match else "FAST"
    except Exception:
        return "FAST"


def route(
    messages: list,
    history,
    pseudo_model: str = "autoconduck",
    tiebreaker=None,
    config=None,
) -> RoutingDecision:
    if config is None:
        from .config import get_config

        config = get_config()
    enabled = [
        entry
        for entry in (getattr(config, "model_list", []) or [])
        if entry.get("enabled", True)
    ]
    user_messages = _user_messages(messages)
    last = user_messages[-1] if user_messages else ""
    text = (
        last.get("content", "")
        if isinstance(last, dict)
        else getattr(last, "content", str(last))
    )
    text = evaluator.clean_routing_text(text)
    match = semantic_router.route(text)
    result = evaluator.score(messages, history, match, pseudo_model, config)
    if len(enabled) <= 1 and result.confidence_band == "ambiguous":
        path = "slow" if match.route == "slow_path" else "fast"
        model = (
            pricing.select_closest(
                pricing.pool_ids(config),
                result.complexity,
                config,
                pseudo_model=pseudo_model,
            )
            if path == "fast"
            else None
        )
        if model:
            from .stats import record_selection

            record_selection(
                result.complexity,
                pricing.target_scaled_cost(result.complexity, pseudo_model, config),
                model,
                config,
            )
        return RoutingDecision(
            path=path,
            confidence_band="ambiguous",
            confidence=result.confidence,
            complexity=result.complexity,
            reason=f"single-model, router-resolved: {path}",
            model=model,
        )
    if result.confidence_band == "ambiguous":
        selection = getattr(config, "selection", config)
        tiebreaker_floor = float(getattr(selection, "tiebreaker_min_complexity", 0.45))
        if pseudo_model.endswith("budget"):
            tiebreaker_floor = float(
                getattr(selection, "budget_tiebreaker_min_complexity", 0.65)
            )
        use_tiebreaker = tiebreaker is not None or (
            bool(getattr(selection, "tiebreaker_enabled", True))
            and result.complexity >= tiebreaker_floor
        )
        if not use_tiebreaker:
            path, complexity = "fast", result.complexity
        else:
            try:
                answer = str(
                    (tiebreaker or _default_tiebreaker)(
                        messages[-1], pseudo_model, config
                    )
                ).upper()
                import re

                match = re.match(r"^(FAST|SLOW)(?:\s+([1-9]))?\s*$", answer)
                digit = int(match.group(2)) if match and match.group(2) else None
                complexity = (
                    0.5 * result.complexity + 0.5 * (digit / 9)
                    if digit is not None
                    else result.complexity
                )
                path = "slow" if match and match.group(1) == "SLOW" else "fast"
            except Exception:
                path, complexity = "fast", result.complexity
        model = (
            pricing.select_closest(
                pricing.pool_ids(config), complexity, config, pseudo_model=pseudo_model
            )
            if path == "fast"
            else None
        )
        if model:
            from .stats import record_selection

            record_selection(
                complexity,
                pricing.target_scaled_cost(complexity, pseudo_model, config),
                model,
                config,
            )
        return RoutingDecision(
            path=path,
            confidence_band="ambiguous",
            confidence=result.confidence,
            complexity=complexity,
            reason="tiebreaker: " + path,
            model=model,
        )
    model = (
        pricing.select_closest(
            pricing.pool_ids(config),
            result.complexity,
            config,
            pseudo_model=pseudo_model,
        )
        if result.path == "fast"
        else None
    )
    if model:
        from .stats import record_selection

        record_selection(
            result.complexity,
            pricing.target_scaled_cost(result.complexity, pseudo_model, config),
            model,
            config,
        )
    return RoutingDecision(
        path=result.path,
        confidence_band=result.confidence_band,
        confidence=result.confidence,
        complexity=result.complexity,
        reason=result.reason,
        model=model,
    )
