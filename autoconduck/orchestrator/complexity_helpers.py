from __future__ import annotations


STAGNATION_MARKER = "<loop-stagnation:true>"


def stagnation_signal(messages: list) -> bool:
    """Allow complexity to be rescored after an executor loop warning."""
    if not isinstance(messages, list):
        return False
    return any(
        isinstance(message, dict)
        and STAGNATION_MARKER in str(message.get("content", ""))
        for message in messages[-6:]
    )
