# Session Lifecycle & Context Guard

## 1. Overview
Coding assistants frequently maintain long sessions (40+ turns). Naive context manipulation breaks upstream prompt caching (e.g. Anthropic prompt caching, OpenAI prefix caching). AutoConduck provides SessionGuard (orchestrator/session_guard.py) to enforce strict prefix immutability and context window ceiling protection.

## 2. Invariants
- **Immutable Prefix Contract**: System prompts and initial user instructions (turns 0 and 1) are guaranteed to remain byte-identical across all turns.
- **80% Context Window Ceiling**: When estimated token volume exceeds 80% of model context window, intelligent compaction is applied to intermediate turns.
- **Fenced Content Preservation**: Compaction preserves code fences and markdown structural headers (#, ##), compacting only verbose tool output logs.
