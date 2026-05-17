from __future__ import annotations

import agentscribe.adapters.utils as utils


def test_utils_package_reexports_shared_helpers() -> None:
    assert set(utils.__all__) == {
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
    }
