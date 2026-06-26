# validation.py
"""Conformance validation for fine-tuning dataset records.

This is the counterpart to :class:`~agentscribe.core.formatter.Formatter`: the
formatter *produces* a record in a target format, and this module *proves* a
record actually follows that format's contract. It is intentionally small and
dependency-free so it can run anywhere — inside e2e tests, a CLI lint step, or a
user's own dataset pipeline.

Each check is structural: it verifies the keys, types, and cross-references a
format requires (e.g. an OpenAI ``tool`` message must point at a ``tool_call``
that was actually announced). It does *not* judge content quality — an Alpaca
record with an empty ``output`` is still a structurally valid Alpaca record.

Usage::

    issues = validate_record(record, "openai_chat")   # [] == conformant
    assert_valid(record, "sharegpt")                  # raises on first problem


Expected record shapes (the authoritative reference)
----------------------------------------------------
These are exactly what ``validate_record`` accepts as correct for each format.

``openai_chat`` — OpenAI fine-tuning / chat completion shape::

    {"messages": [
        {"role": "system",    "content": "..."},                       # optional
        {"role": "user",      "content": "..."},
        {"role": "assistant", "content": null,                         # text may be null when calling a tool
         "tool_calls": [{"id": "call_1", "type": "function",
                         "function": {"name": "multiply",
                                      "arguments": "{\\"a\\": 6, \\"b\\": 7}"}}]},  # arguments is a JSON *string*
        {"role": "tool", "tool_call_id": "call_1", "content": "42"},    # tool_call_id MUST match an announced call
        {"role": "assistant", "content": "It's 42."}]}
    # roles ⊆ {system, user, assistant, tool}; every tool message links to a prior assistant tool_call id.

``sharegpt`` — LLaMA-Factory / Axolotl shape::

    {"system": "...",                                                   # optional; system is its OWN key, never a turn
     "conversations": [
        {"from": "human",         "value": "..."},
        {"from": "function_call", "value": "{\\"name\\": \\"multiply\\", \\"arguments\\": {...}}"},  # JSON w/ a name
        {"from": "observation",   "value": "42"},
        {"from": "gpt",           "value": "..."}]}
    # from ∈ {human, gpt, system, function_call, observation, tool}; 'system' must NOT appear as a turn.

``alpaca`` — instruction-tuning shape (text only; tool turns are not represented)::

    {"instruction": "...", "input": "", "output": "...",
     "system": "...",                                                   # optional
     "history": [["user turn", "assistant turn"], ...]}                 # optional list of [str, str] pairs

``prompt_completion`` — legacy single-turn shape::

    {"prompt": "...", "completion": "..."}

``preference`` — TRL DPO conversational shape::

    {"prompt":   [ {"role": "user", "content": "..."}, ... ],           # context as message objects
     "chosen":   [ {"role": "assistant", "content": "..."} ],
     "rejected": [ {"role": "assistant", "content": "..."} ]}           # filled by a judge / default
"""

from __future__ import annotations

import json
from typing import Any

SUPPORTED_FORMATS = (
    "openai_chat",
    "sharegpt",
    "alpaca",
    "prompt_completion",
    "preference",
)


class FormatValidationError(ValueError):
    """Raised by :func:`assert_valid` when a record violates its format."""


# --------------------------------------------------------------------------- #
# Small shared helpers
# --------------------------------------------------------------------------- #
def _is_str(value: Any) -> bool:
    return isinstance(value, str)


def _json_loads(value: Any) -> tuple[bool, Any]:
    """Return (ok, parsed) for a JSON string; (False, None) otherwise."""
    if not isinstance(value, str):
        return False, None
    try:
        return True, json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return False, None


# --------------------------------------------------------------------------- #
# Per-format validators — each returns a list of human-readable issue strings.
# --------------------------------------------------------------------------- #
_OPENAI_ROLES = {"system", "user", "assistant", "tool"}


def _validate_openai_chat(record: dict[str, Any]) -> list[str]:
    """See the module docstring for the full ``openai_chat`` shape."""
    issues: list[str] = []
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        return ["openai_chat: 'messages' must be a non-empty list"]

    announced_call_ids: set[str] = set()
    for idx, m in enumerate(messages):
        if not isinstance(m, dict):
            issues.append(f"openai_chat: message[{idx}] is not an object")
            continue
        role = m.get("role")
        if role not in _OPENAI_ROLES:
            issues.append(f"openai_chat: message[{idx}] has invalid role {role!r}")
            continue

        if role == "assistant":
            for c_idx, call in enumerate(m.get("tool_calls") or []):
                loc = f"openai_chat: message[{idx}].tool_calls[{c_idx}]"
                if not isinstance(call, dict):
                    issues.append(f"{loc} is not an object")
                    continue
                if call.get("type") != "function":
                    issues.append(f"{loc} missing type='function'")
                call_id = call.get("id")
                if not _is_str(call_id) or not call_id:
                    issues.append(f"{loc} missing string 'id'")
                else:
                    announced_call_ids.add(call_id)
                fn = call.get("function")
                if not isinstance(fn, dict) or not _is_str(fn.get("name")) or not fn.get("name"):
                    issues.append(f"{loc}.function missing 'name'")
                else:
                    ok, _ = _json_loads(fn.get("arguments"))
                    if not ok:
                        issues.append(f"{loc}.function.arguments is not a JSON string")

        elif role == "tool":
            call_id = m.get("tool_call_id")
            if not _is_str(call_id) or not call_id:
                issues.append(f"openai_chat: message[{idx}] (tool) missing 'tool_call_id'")
            elif call_id not in announced_call_ids:
                issues.append(
                    f"openai_chat: message[{idx}] (tool) references unknown tool_call_id {call_id!r}"
                )
            if not _is_str(m.get("content")):
                issues.append(f"openai_chat: message[{idx}] (tool) content must be a string")
    return issues


_SHAREGPT_FROMS = {"human", "gpt", "system", "function_call", "observation", "tool"}


def _validate_sharegpt(record: dict[str, Any]) -> list[str]:
    """See the module docstring for the full ``sharegpt`` shape."""
    issues: list[str] = []
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        return ["sharegpt: 'conversations' must be a non-empty list"]
    if "system" in record and not _is_str(record["system"]):
        issues.append("sharegpt: top-level 'system' must be a string")

    for idx, turn in enumerate(conversations):
        if not isinstance(turn, dict):
            issues.append(f"sharegpt: conversations[{idx}] is not an object")
            continue
        sender = turn.get("from")
        if sender not in _SHAREGPT_FROMS:
            issues.append(f"sharegpt: conversations[{idx}] has invalid 'from' {sender!r}")
        if sender == "system":
            issues.append(
                f"sharegpt: conversations[{idx}] uses 'system' turn; system belongs in the top-level key"
            )
        if not _is_str(turn.get("value")):
            issues.append(f"sharegpt: conversations[{idx}] 'value' must be a string")
        elif sender == "function_call":
            ok, parsed = _json_loads(turn.get("value"))
            if not ok or not isinstance(parsed, dict) or "name" not in parsed:
                issues.append(
                    f"sharegpt: conversations[{idx}] function_call value must be JSON with a 'name'"
                )
    return issues


def _validate_alpaca(record: dict[str, Any]) -> list[str]:
    """See the module docstring for the full ``alpaca`` shape."""
    issues: list[str] = []
    for key in ("instruction", "input", "output"):
        if key not in record:
            issues.append(f"alpaca: missing required key {key!r}")
        elif not _is_str(record[key]):
            issues.append(f"alpaca: {key!r} must be a string")
    if "system" in record and not _is_str(record["system"]):
        issues.append("alpaca: optional 'system' must be a string")
    if "history" in record:
        history = record["history"]
        if not isinstance(history, list):
            issues.append("alpaca: optional 'history' must be a list")
        else:
            for idx, pair in enumerate(history):
                if not (isinstance(pair, (list, tuple)) and len(pair) == 2 and all(_is_str(p) for p in pair)):
                    issues.append(f"alpaca: history[{idx}] must be a [user, assistant] string pair")
    return issues


def _validate_prompt_completion(record: dict[str, Any]) -> list[str]:
    """See the module docstring for the full ``prompt_completion`` shape."""
    issues: list[str] = []
    for key in ("prompt", "completion"):
        if key not in record:
            issues.append(f"prompt_completion: missing required key {key!r}")
        elif not _is_str(record[key]):
            issues.append(f"prompt_completion: {key!r} must be a string")
    return issues


def _validate_preference(record: dict[str, Any]) -> list[str]:
    """See the module docstring for the full ``preference`` shape."""
    issues: list[str] = []
    for key in ("prompt", "chosen", "rejected"):
        if key not in record:
            issues.append(f"preference: missing required key {key!r}")
        elif not isinstance(record[key], list):
            issues.append(f"preference: {key!r} must be a list of message objects")
    for key in ("chosen", "rejected"):
        for idx, msg in enumerate(record.get(key, []) or []):
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                issues.append(f"preference: {key}[{idx}] must be a message object with role/content")
    return issues


_VALIDATORS = {
    "openai_chat": _validate_openai_chat,
    "sharegpt": _validate_sharegpt,
    "alpaca": _validate_alpaca,
    "prompt_completion": _validate_prompt_completion,
    "preference": _validate_preference,
}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def validate_record(record: Any, format_name: str) -> list[str]:
    """Return a list of conformance issues for ``record`` (empty == valid)."""
    if format_name not in _VALIDATORS:
        raise ValueError(
            f"Unsupported format: {format_name!r}. Choose from {list(SUPPORTED_FORMATS)}."
        )
    if not isinstance(record, dict):
        return [f"{format_name}: record must be a JSON object, got {type(record).__name__}"]
    return _VALIDATORS[format_name](record)


def is_valid(record: Any, format_name: str) -> bool:
    """Convenience boolean wrapper around :func:`validate_record`."""
    return not validate_record(record, format_name)


def assert_valid(record: Any, format_name: str) -> None:
    """Raise :class:`FormatValidationError` if ``record`` is non-conformant."""
    issues = validate_record(record, format_name)
    if issues:
        raise FormatValidationError(
            f"{format_name} record failed validation:\n  - " + "\n  - ".join(issues)
        )


__all__ = [
    "SUPPORTED_FORMATS",
    "FormatValidationError",
    "validate_record",
    "is_valid",
    "assert_valid",
]
