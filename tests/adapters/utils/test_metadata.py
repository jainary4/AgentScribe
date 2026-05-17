from __future__ import annotations

from dataclasses import dataclass

from agentscribe.adapters.utils.metadata import build_metadata, resolve_identifier, resolve_value, serialize_object_list


@dataclass
class Item:
    name: str


def test_resolve_value_handles_callable_string_and_fallback_sequence() -> None:
    source = {"primary": None, "secondary": "value", "name": "item"}

    assert resolve_value(source, lambda value: value["name"].upper()) == "ITEM"
    assert resolve_value(source, "name") == "item"
    assert resolve_value(source, ("primary", "secondary")) is None
    assert resolve_value(source, ("missing", "secondary")) == "value"
    assert resolve_value(source, "missing", default="fallback") == "fallback"


def test_resolve_identifier_stringifies_found_values_and_returns_none_for_missing() -> None:
    assert resolve_identifier({"id": 123}, "id") == "123"
    assert resolve_identifier({}, "id") is None


def test_serialize_object_list_handles_scalars_mappings_and_dataclasses() -> None:
    assert serialize_object_list([{"name": "dict"}, Item("data"), "raw"]) == [
        {"name": "dict"},
        {"name": "data"},
        "raw",
    ]


def test_build_metadata_compacts_empty_values() -> None:
    metadata = build_metadata(
        {"name": "agent", "empty": None},
        fields={"name": "name", "missing": "empty", "derived": lambda source: source["name"].upper()},
    )

    assert metadata == {"name": "agent", "derived": "AGENT"}
