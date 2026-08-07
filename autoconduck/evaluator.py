from __future__ import annotations

import math
import re
from typing import Any

try:
    import tiktoken

    _enc = tiktoken.get_encoding("cl100k_base")

    def _token_len(text: str) -> int:
        try:
            return len(_enc.encode(text))
        except Exception:
            return max(1, len(text) // 4)

except ImportError:

    def _token_len(text: str) -> int:
        return max(1, len(text) // 4)


STACK_RE = re.compile(r"(Traceback|at\s+\w+\.\w+\(|UnhandledPromiseRejection|Error:|Exception:)", re.I)

HYSTERESIS_THRESHOLD = 0.80
HYSTERESIS_CLAMP = 0.50

# Weights for weighted sum
WEIGHTS = {
    "token_len": 0.35,
    "code_ratio": 0.25,
    "file_refs": 0.20,
    "imperative": 0.10,
    "question_depth": 0.10,
}

IMPERATIVE_KEYWORDS = ["implement", "create", "analyze", "build", "refactor", "migrate", "design"]
FILE_REF_RE = re.compile(r"[\w/\-.]+\.(py|ts|js|go|rs|md|java|cpp|yaml|json|toml)\b", re.I)
CODE_FENCE_RE = re.compile(r"```")


def _extract_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                elif "content" in item:
                    parts.append(str(item["content"]))
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def score(last_message: Any, prev_state: Any | None = None) -> float:
    """
    Compute T_i in [0,1] for the last message only.
    prev_state: TurnState | None (reads used_reasoning_tier)
    Pure-ish CPU function, <1ms.
    """
    try:
        content = getattr(last_message, "content", last_message)
        # last_message may be dict
        if isinstance(last_message, dict):
            content = last_message.get("content", "")
        text = _extract_text(content)
        if not text:
            return 0.0

        # feature: token_len normalized 0-1 (cap at 4000 tokens -> 1.0)
        t_len = _token_len(text)
        token_len_f = min(1.0, t_len / 4000.0)

        # code_ratio: estimate via code fence or file refs heuristic + code-like chars ratio
        total_chars = max(1, len(text))
        fences = len(CODE_FENCE_RE.findall(text))
        # heuristic: characters inside fences approximated; use code_ratio as weighted blend
        if fences >= 2:
            # there is at least one code block
            # rough: assume 30% code per fence pair + file ref boost
            code_ratio_f = min(1.0, 0.5 + fences * 0.1)
        else:
            # fallback: ratio of code-like tokens (indentation, braces, keywords)
            code_like = len(re.findall(r"[{};=]|def |class |import |function |const |let ", text))
            code_ratio_f = min(1.0, code_like / 20.0)

        file_refs = len(FILE_REF_RE.findall(text))
        file_refs_f = min(1.0, file_refs / 5.0)

        lower = text.lower()
        imperative_f = 0.2 if any(kw in lower for kw in IMPERATIVE_KEYWORDS) else 0.0

        question_depth_f = min(0.3, 0.15 * text.count("?"))

        features = {
            "token_len": token_len_f,
            "code_ratio": code_ratio_f,
            "file_refs": file_refs_f,
            "imperative": imperative_f,
            "question_depth": question_depth_f,
        }
        raw = sum(features[k] * WEIGHTS[k] for k in WEIGHTS)
        # sigmoid sharp around 0.5
        t = _sigmoid((raw - 0.5) * 6)
        t = max(0.0, min(1.0, t))

        # Stack trace boost
        has_stack = bool(STACK_RE.search(text))
        if has_stack:
            t = min(1.0, t + 0.25)

        # Hysteresis cooldown
        if prev_state is not None:
            used_reasoning = getattr(prev_state, "used_reasoning_tier", False)
            if used_reasoning and not has_stack:
                # clamp to 0.50
                t = min(t, HYSTERESIS_CLAMP)

        return float(max(0.0, min(1.0, t)))
    except Exception:
        return 0.5
