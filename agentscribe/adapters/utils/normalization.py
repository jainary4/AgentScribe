"""Normalization helpers for optional, duck-typed framework adapters."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID

from agentscribe.core.canonical import CanonicalInteraction, CanonicalMessage, MessageRole


MISSING = object()
ROLE_ALIASES = {
    "ai": "assistant",
    "bot": "assistant",
    "function": "tool_response",
    "function_call": "tool_call",
    "gpt": "assistant",
    "human": "user",
    "observation": "tool_response",
    "tool": "tool_response",
}
CANONICAL_ROLES = {"system", "user", "assistant", "tool_call", "tool_response"}


def json_ready(value: Any) -> Any:
    """Convert common Python and framework objects into JSON-safe values."""

    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [json_ready(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return json_ready(asdict(value))
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (Decimal, UUID, Path)):
        return str(value)
    if hasattr(value, "model_dump"):
        try:
            return json_ready(value.model_dump(mode="json"))
        except Exception:
            try:
                return json_ready(value.model_dump())
            except Exception:
                return str(value)
    if hasattr(value, "to_dict"):
        try:
            return json_ready(value.to_dict())
        except Exception:
            pass
    # Last resort: stringify anything not JSON-native so capture never crashes.
    if value is not None and not isinstance(value, (str, int, float, bool)):
        return str(value)
    return value


def coerce_text(value: Any) -> str:
    """Coerce arbitrary content into canonical message text."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(json_ready(value), ensure_ascii=False, default=str)


def normalize_role(role: Any, *, default: MessageRole = "assistant") -> MessageRole:
    """Normalize framework-specific role names into canonical roles."""

    if role is None:
        return default
    role_text = str(role).strip().lower()
    role_text = ROLE_ALIASES.get(role_text, role_text)
    if role_text in CANONICAL_ROLES:
        return role_text  # type: ignore[return-value]
    return default


def _metadata_property(key: str, default_factory: Any = None) -> property:
    def getter(interaction: CanonicalInteraction) -> Any:
        if default_factory is list:
            return interaction.metadata.setdefault(key, [])
        if default_factory is dict:
            return interaction.metadata.setdefault(key, {})
        return interaction.metadata.get(key)

    def setter(interaction: CanonicalInteraction, value: Any) -> None:
        if value is None:
            interaction.metadata.pop(key, None)
        else:
            interaction.metadata[key] = json_ready(value)

    return property(getter, setter)


def _dict_or_wrapped(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    parsed = parse_jsonish(value)
    if isinstance(parsed, Mapping):
        return dict(parsed)
    return {"value": json_ready(parsed)}


def _interaction_add_message(self: CanonicalInteraction, role: MessageRole, content: Any = "", **kwargs: Any) -> CanonicalMessage:
    message = CanonicalMessage(
        role=normalize_role(role),
        content=coerce_text(content),
        tool_name=kwargs.get("tool_name"),
        tool_args=_dict_or_wrapped(kwargs.get("tool_args")),
        tool_result=coerce_text(kwargs.get("tool_result")) if kwargs.get("tool_result") is not None else None,
    )
    self.messages.append(message)
    return message


def _install_interaction_metadata_views() -> None:
    if not hasattr(CanonicalInteraction, "add_message"):
        setattr(CanonicalInteraction, "add_message", _interaction_add_message)
    for key, default_factory in {
        "agent": dict,
        "extra": dict,
        "instantiation": dict,
        "model": None,
        "provider": None,
        "run_id": None,
        "spans": list,
        "thread_id": None,
        "token_usage": dict,
        "tools": list,
        "trace_id": None,
    }.items():
        if not hasattr(CanonicalInteraction, key):
            setattr(CanonicalInteraction, key, _metadata_property(key, default_factory))


_install_interaction_metadata_views()


def get_value(source: Any, *names: str, default: Any = None) -> Any:
    """Read the first matching key or attribute from a mapping or object."""

    if source is None:
        return default
    for name in names:
        if isinstance(source, Mapping) and name in source:
            return source[name]
        if hasattr(source, name):
            return getattr(source, name)
    return default


def get_nested(source: Any, *path: str, default: Any = None) -> Any:
    """Read a nested key or attribute path without assuming a concrete type."""

    current = source
    for name in path:
        current = get_value(current, name, default=MISSING)
        if current is MISSING:
            return default
    return current


def object_to_dict(value: Any) -> dict[str, Any]:
    """Best-effort conversion of framework objects into a dictionary."""

    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    for method_name in ("model_dump", "dict", "to_dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                payload = method(mode="json") if method_name == "model_dump" else method()
            except Exception:
                try:
                    payload = method()
                except Exception:
                    continue
            if isinstance(payload, Mapping):
                return dict(payload)
    instance_dict = getattr(value, "__dict__", None)
    if isinstance(instance_dict, Mapping):
        return {str(key): item for key, item in instance_dict.items() if not str(key).startswith("_")}
    return {}


def as_list(value: Any) -> list[Any]:
    """Return value as a list, treating None as empty and strings as scalar."""

    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return [value]


def parse_jsonish(value: Any) -> Any:
    """Parse JSON strings when possible; otherwise return the original value."""

    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{\"":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def compact_dict(data: Mapping[str, Any]) -> dict[str, Any]:
    """Drop empty optional values and JSON-normalize the rest."""

    return {
        str(key): json_ready(value)
        for key, value in data.items()
        if value is not None and value != {} and value != []
    }


def infer_message_role(message: Any, *, default_role: MessageRole = "assistant") -> MessageRole:
    """Infer a canonical role from framework message fields and class names."""

    explicit_role = get_value(message, "role", "from", "author", default=None)
    if explicit_role is not None:
        return normalize_role(explicit_role, default=default_role)

    source = get_value(message, "source", "sender", "speaker", default=None)
    if source is not None and str(source).strip().lower() in {"user", "human", "system", "assistant", "ai"}:
        return normalize_role(source, default=default_role)

    message_type = str(get_value(message, "type", "event_type", default=message.__class__.__name__)).lower()
    if "toolcallrequest" in message_type or "tool_call_request" in message_type or "function_call" in message_type:
        return "tool_call"
    if "toolcallexecution" in message_type or "tool_call_execution" in message_type or "tool_result" in message_type:
        return "tool_response"
    if "system" in message_type:
        return "system"
    if "user" in message_type or "human" in message_type:
        return "user"
    return default_role


def message_to_canonical(
    message: Any,
    *,
    default_role: MessageRole = "assistant",
    metadata: Mapping[str, Any] | None = None,
    preserve_raw: bool = False,
) -> CanonicalMessage:
    """Convert a framework message object or dict into a canonical message."""

    if isinstance(message, CanonicalMessage):
        return message

    message_mapping = object_to_dict(message)
    role = infer_message_role(message, default_role=default_role)
    content = get_value(message, "content", "value", "text", "chat_message", "message", "raw", "output", default="")
    tool_name = get_value(message, "tool_name", "function_name", default=None)
    if role in {"tool_call", "tool_response"} and tool_name is None:
        tool_name = get_value(message, "name", default=None)
    tool_args = parse_jsonish(get_value(message, "tool_args", "arguments", "args", "input", default=None))
    tool_result = get_value(message, "tool_result", "result", "output", default=None)

    if not content and role == "tool_call":
        content = {"tool_name": tool_name, "tool_args": tool_args}
    if not content and role == "tool_response":
        content = tool_result

    canonical = CanonicalMessage(
        role=role,
        content=coerce_text(content),
        tool_name=str(tool_name) if tool_name is not None else None,
        tool_args=_dict_or_wrapped(tool_args),
        tool_result=coerce_text(tool_result) if tool_result is not None else None,
    )
    message_metadata = compact_dict(
        {
            "source": get_value(message, "source", "sender", "speaker", default=None),
            "type": get_value(message, "type", "event_type", default=message.__class__.__name__),
            "tool_call_id": get_value(message, "tool_call_id", "call_id", "id", default=None),
            "tool_calls": get_value(message, "tool_calls", "function_calls", default=None),
            "function_call": get_value(message, "function_call", default=None),
            "token_usage": object_to_dict(get_value(message, "token_usage", "usage", "models_usage", default={})),
            **dict(metadata or {}),
        }
    )
    if message_metadata:
        setattr(canonical, "metadata", message_metadata)
    if preserve_raw and message_mapping:
        setattr(canonical, "raw", json_ready(message_mapping))
    return canonical


def function_call_to_tool_call(call: Any) -> dict[str, Any]:
    """Convert a function or tool call object into OpenAI-compatible shape."""

    call_mapping = object_to_dict(call)
    call_id = get_value(call, "id", "call_id", "tool_call_id", default=None)
    name = get_value(call, "name", "tool_name", "function_name", default="")
    arguments = parse_jsonish(get_value(call, "arguments", "args", "tool_args", default={}))
    serialized_arguments = arguments if isinstance(arguments, str) else json.dumps(json_ready(arguments), ensure_ascii=False)
    result = {"type": "function", "function": {"name": str(name), "arguments": serialized_arguments}}
    if call_id is not None:
        result["id"] = str(call_id)
    if call_mapping:
        result["metadata"] = json_ready(call_mapping)
    return result


def tool_call_message(
    tool_name: str | None,
    tool_args: Any = None,
    *,
    tool_call_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> CanonicalMessage:
    """Create a canonical tool-call message using compact canonical fields."""

    parsed_args = parse_jsonish(tool_args)
    canonical_args = _dict_or_wrapped(parsed_args) or {}
    aux = compact_dict({"tool_call_id": tool_call_id, "metadata": dict(metadata or {})})
    if aux:
        canonical_args.setdefault("_agentscribe", {}).update(aux)
    return CanonicalMessage(
        role="tool_call",
        content=coerce_text({"tool_name": tool_name, "tool_args": parsed_args}),
        tool_name=tool_name,
        tool_args=canonical_args,
    )


def tool_response_message(
    tool_name: str | None,
    tool_result: Any = None,
    *,
    tool_call_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> CanonicalMessage:
    """Create a canonical tool-response message using compact canonical fields."""

    message = CanonicalMessage(
        role="tool_response",
        content=coerce_text(tool_result),
        tool_name=tool_name,
        tool_result=coerce_text(tool_result) if tool_result is not None else None,
    )
    aux = compact_dict({"tool_call_id": tool_call_id, "metadata": dict(metadata or {})})
    if aux:
        setattr(message, "metadata", aux)
    return message


def interaction_from_messages(
    messages: Iterable[Any],
    *,
    source_framework: str,
    default_role: MessageRole = "assistant",
    metadata: Mapping[str, Any] | None = None,
    **interaction_fields: Any,
) -> CanonicalInteraction:
    """Create an interaction from framework-native message objects."""

    session_id = interaction_fields.pop("session_id", None)
    interaction_id = interaction_fields.pop("id", None)
    timestamp = interaction_fields.pop("timestamp", None)
    interaction = CanonicalInteraction(
        id=str(interaction_id) if interaction_id is not None else CanonicalInteraction().id,
        source_framework=source_framework,
        session_id=str(session_id) if session_id is not None else None,
        timestamp=str(timestamp) if timestamp is not None else CanonicalInteraction().timestamp,
        metadata=dict(metadata or {}),
    )
    interaction.metadata.update(compact_dict(interaction_fields))
    for message in messages:
        interaction.messages.append(message_to_canonical(message, default_role=default_role))
    return interaction


def append_unique_message(interaction: CanonicalInteraction, message: CanonicalMessage) -> None:
    """Append a message unless an equivalent role/content/tool payload exists."""

    signature = (message.role, message.content, message.tool_name, json.dumps(json_ready(message.tool_args), sort_keys=True, default=str))
    for existing_message in interaction.messages:
        existing_signature = (
            existing_message.role,
            existing_message.content,
            existing_message.tool_name,
            json.dumps(json_ready(existing_message.tool_args), sort_keys=True, default=str),
        )
        if existing_signature == signature:
            return
    interaction.messages.append(message)


__all__ = [
    "append_unique_message",
    "as_list",
    "coerce_text",
    "compact_dict",
    "function_call_to_tool_call",
    "get_nested",
    "get_value",
    "interaction_from_messages",
    "json_ready",
    "message_to_canonical",
    "normalize_role",
    "object_to_dict",
    "parse_jsonish",
    "tool_call_message",
    "tool_response_message",
]