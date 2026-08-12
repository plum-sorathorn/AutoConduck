from __future__ import annotations

import re



def _tokens(text: str) -> set[str]:
    """Tokenise text into a lowercase word set (shared with semantic_router)."""
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def _intent_drift(messages: list) -> float:
    """Jaccard distance between first and last user messages.

    Returns 0.0 when there is only one user turn (no drift possible).
    High value → conversation has moved far from original intent → scope expansion.
    """
    user_msgs = [
        m for m in messages
        if isinstance(m, dict) and m.get("role") == "user"
        and "<system-reminder>" not in str(m.get("content", ""))
    ]
    if len(user_msgs) < 2:
        return 0.0
    first = _tokens(str(user_msgs[0].get("content", "")))
    last = _tokens(str(user_msgs[-1].get("content", "")))
    if not first or not last:
        return 0.0
    union = first | last
    jaccard = len(first & last) / len(union)
    return 1.0 - jaccard
def _first_user_complexity(messages: list, config=None) -> float:
    """Compute complexity of the *first* user message in the conversation.

    Used to prevent the tool-loop fast-path from suppressing a SLOW routing
    decision when the original request was genuinely complex.  Returns 0.0 if
    no user message is found.
    """
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "user")
        content = str(msg.get("content", ""))
        if role == "user" and "<system-reminder>" not in content and content.strip():
            from .complexity import complexity_of
            return complexity_of(content, config)
    return 0.0
def _last(messages: list) -> str:
    if not isinstance(messages, list) or not messages:
        return ""
    for item in reversed(messages):
        if not isinstance(item, dict) or item.get("role", "user") == "user":
            content = (
                item.get("content", "")
                if isinstance(item, dict)
                else getattr(item, "content", item)
            )
            if "<system-reminder>" not in str(content):
                return str(content or "")
                from .complexity import clean_routing_text
                return clean_routing_text(content)
    return ""


def _routing_text(messages: list) -> str:
    user_text = _last(messages)
    tool_text = ""
    if isinstance(messages, list) and messages:
        last = messages[-1]
        if isinstance(last, dict) and last.get("role") in ("tool", "function"):
            tool_text = str(last.get("content", ""))
    return f"{user_text}\n{tool_text}".strip()


def _context_boost(messages: list) -> float:
    """Compute an additive context-aware complexity boost ∈ [0, 0.20].

    Three sub-signals, all O(m) over messages (typically <20):
      conversation_depth  — how many user turns have accumulated
      tool_chain_length   — how many tool calls/results have fired
      intent_drift        — Jaccard distance between first and last user message

    The boost is capped at 0.20 so it cannot dominate the base complexity score.
    It is applied in score() *after* complexity_of() so complexity_of() stays
    pure-text and independently testable.
    """
    if not isinstance(messages, list) or not messages:
        return 0.0

    user_turn_count = sum(
        1 for m in messages
        if isinstance(m, dict) and m.get("role") == "user"
        and "<system-reminder>" not in str(m.get("content", ""))
    )
    # 0.0 on first turn, 1.0 at 10+ turns
    conversation_depth = min(1.0, max(0.0, (user_turn_count - 1) / 10))

    tool_call_count = sum(
        1 for m in messages
        if isinstance(m, dict) and (
            m.get("role") in ("tool", "function") or "tool_calls" in m
        )
    )
    # 0.0 at 0 calls, 1.0 at 8+ calls
    tool_chain_length = min(1.0, tool_call_count / 8)

    drift = _intent_drift(messages)

    boost = 0.08 * conversation_depth + 0.08 * tool_chain_length + 0.04 * drift
    return min(0.20, boost)
