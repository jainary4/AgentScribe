from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

from agentscribe.adapters.utils.normalization import (
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
from agentscribe.core.canonical import CanonicalInteraction, CanonicalMessage


@dataclass
class Payload:
    value: int


def test_json_ready_converts_common_non_json_types() -> None:
    value = {
        "date": date(2026, 1, 2),
        "datetime": datetime(2026, 1, 2, 3, 4, 5),
        "decimal": Decimal("1.5"),
        "path": Path("file.txt"),
        "uuid": UUID("00000000-0000-0000-0000-000000000001"),
        "payload": Payload(1),
    }

    assert json_ready(value) == {
        "date": "2026-01-02",
        "datetime": "2026-01-02T03:04:05",
        "decimal": "1.5",
        "path": "file.txt",
        "uuid": "00000000-0000-0000-0000-000000000001",
        "payload": {"value": 1},
    }


def test_coerce_text_roles_lists_jsonish_and_compaction_boundaries() -> None:
    assert coerce_text(None) == ""
    assert coerce_text({"a": 1}) == '{"a": 1}'
    assert normalize_role("human") == "user"
    assert normalize_role("unknown", default="system") == "system"
    assert as_list(None) == []
    assert as_list("x") == ["x"]
    assert parse_jsonish('{"a":1}') == {"a": 1}
    assert parse_jsonish("{bad") == "{bad"
    assert compact_dict({"a": 1, "none": None, "empty": [], "dict": {}}) == {"a": 1}


def test_get_value_get_nested_and_object_to_dict_support_mapping_objects_and_dataclasses() -> None:
    obj = SimpleNamespace(name="object", nested={"value": 3})

    assert get_value({"name": "dict"}, "name") == "dict"
    assert get_value(obj, "name") == "object"
    assert get_value(None, "missing", default="fallback") == "fallback"
    assert get_nested(obj, "nested", "value") == 3
    assert get_nested(obj, "nested", "missing", default="fallback") == "fallback"
    assert object_to_dict(Payload(2)) == {"value": 2}
    assert object_to_dict(SimpleNamespace(public=1, _private=2)) == {"public": 1}


def test_message_to_canonical_normalizes_roles_tools_metadata_and_raw_payload() -> None:
    message = message_to_canonical(
        {
            "type": "ToolCallRequestEvent",
            "name": "lookup",
            "arguments": '{"q":"x"}',
            "id": "call-1",
        },
        preserve_raw=True,
    )

    assert message.role == "tool_call"
    assert message.tool_name == "lookup"
    assert message.tool_args == {"q": "x"}
    assert message.metadata["tool_call_id"] == "call-1"
    assert message.raw["name"] == "lookup"


def test_function_call_and_tool_message_builders_preserve_ids_and_metadata() -> None:
    tool_call = function_call_to_tool_call({"id": "call-1", "name": "lookup", "arguments": {"q": "x"}})
    call_message = tool_call_message("lookup", {"q": "x"}, tool_call_id="call-1", metadata={"duration": 1})
    response_message = tool_response_message("lookup", {"ok": True}, tool_call_id="call-1", metadata={"is_error": False})

    assert tool_call["function"]["arguments"] == '{"q": "x"}'
    assert call_message.tool_args["_agentscribe"]["tool_call_id"] == "call-1"
    assert call_message.tool_args["_agentscribe"]["metadata"] == {"duration": 1}
    assert response_message.metadata["tool_call_id"] == "call-1"
    assert response_message.tool_result == '{"ok": true}'


def test_interaction_from_messages_installs_metadata_views_and_append_unique_deduplicates() -> None:
    interaction = interaction_from_messages(
        [{"role": "user", "content": "Q"}],
        source_framework="unit",
        session_id=123,
        run_id="run-1",
    )
    duplicate = CanonicalMessage(role="user", content="Q")
    new_message = CanonicalMessage(role="assistant", content="A")

    append_unique_message(interaction, duplicate)
    append_unique_message(interaction, new_message)
    interaction.model = "model-a"

    assert interaction.session_id == "123"
    assert interaction.run_id == "run-1"
    assert interaction.model == "model-a"
    assert [(message.role, message.content) for message in interaction.messages] == [("user", "Q"), ("assistant", "A")]
