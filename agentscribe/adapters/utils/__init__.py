"""Shared adapter utility package.

This package keeps adapter-only helpers grouped by concern so framework
packages can share normalization, collection, and dispatch code without
growing one monolithic module.
"""

from .collector import AdapterError, InteractionCollector
from .metadata import build_metadata, resolve_identifier, resolve_value, serialize_object_list
from .normalization import (
    append_unique_message,
    as_list,
    coerce_text,
    compact_dict,
    function_call_to_tool_call,
    get_nested,
    get_value,
    interaction_from_messages,
    json_ready,
    message_to_canonical,
    normalize_role,
    object_to_dict,
    parse_jsonish,
    tool_call_message,
    tool_response_message,
)

__all__ = [
    "AdapterError",
    "InteractionCollector",
    "append_unique_message",
    "as_list",
    "build_metadata",
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
    "resolve_identifier",
    "resolve_value",
    "serialize_object_list",
    "tool_call_message",
    "tool_response_message",
]