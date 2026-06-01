from __future__ import annotations

import json

import pytest

from agentscribe.core.canonical import CanonicalInteraction, CanonicalMessage
from agentscribe.core.formatter import (
    Formatter,
    FormatOptions,
    FormatValidationError,
    OpenAIChatFormatter,
    available_formats,
    format_messages,
)


def _interaction(messages):
    interaction = CanonicalInteraction(source_framework="test")
    interaction.messages = messages
    return interaction


def _tool_exchange():
    """A canonical tool exchange: user -> assistant+tool_call -> tool_response."""
    return _interaction(
        [
            CanonicalMessage(role="user", content="Weather in Toronto?"),
            CanonicalMessage(role="assistant", content=""),
            CanonicalMessage(
                role="tool_call",
                content="",
                tool_name="web_search",
                tool_args={"query": "weather Toronto", "_agentscribe": {"tool_call_id": "call_A"}},
            ),
            CanonicalMessage(role="tool_response", content="12C and sunny", tool_name="web_search",
                             tool_result="12C and sunny"),
            CanonicalMessage(role="assistant", content="It's 12C and sunny."),
        ]
    )


def test_registry_exposes_all_shipped_formats() -> None:
    assert available_formats() == [
        "openai_chat",
        "sharegpt",
        "alpaca",
        "prompt_completion",
        "preference",
        "chatml",
        "canonical",
    ]


def test_openai_chat_produces_structured_linked_tool_calls() -> None:
    record = Formatter("openai_chat", strict=True).format_single(_tool_exchange())
    messages = record["messages"]

    assert {m["role"] for m in messages} <= {"system", "user", "assistant", "tool"}
    tool_calls = [tc for m in messages if m["role"] == "assistant" for tc in m.get("tool_calls", [])]
    assert [tc["id"] for tc in tool_calls] == ["call_A"]
    assert tool_calls[0]["function"]["name"] == "web_search"
    json.loads(tool_calls[0]["function"]["arguments"])  # arguments are valid JSON

    tool_messages = [m for m in messages if m["role"] == "tool"]
    assert len(tool_messages) == 1 and tool_messages[0]["tool_call_id"] == "call_A"


def test_openai_chat_drops_orphan_tool_response_without_synthetic_id() -> None:
    # A tool_response with no preceding tool_call cannot be spec-valid and must
    # be dropped rather than emitted with a minted id.
    interaction = _interaction(
        [
            CanonicalMessage(role="user", content="hi"),
            CanonicalMessage(role="tool_response", content="stale", tool_name="web_search"),
            CanonicalMessage(role="assistant", content="hello"),
        ]
    )
    record = Formatter("openai_chat", strict=True).format_single(interaction)
    assert [m["role"] for m in record["messages"]] == ["user", "assistant"]


def test_validate_rejects_dangling_tool_message() -> None:
    bad = {"messages": [
        {"role": "user", "content": "x"},
        {"role": "tool", "tool_call_id": "ghost", "content": "y"},
    ]}
    with pytest.raises(FormatValidationError):
        OpenAIChatFormatter(FormatOptions()).validate(bad)


def test_validate_rejects_invalid_role() -> None:
    bad = {"messages": [{"role": "tool_call", "content": "{}"}]}
    with pytest.raises(FormatValidationError):
        OpenAIChatFormatter(FormatOptions()).validate(bad)


def test_sharegpt_separates_system_and_emits_function_call() -> None:
    record = Formatter("sharegpt").format_single(_tool_exchange())
    froms = [c["from"] for c in record["conversations"]]
    assert "system" not in froms
    assert "function_call" in froms and "observation" in froms
    function_call = next(c for c in record["conversations"] if c["from"] == "function_call")
    assert json.loads(function_call["value"])["name"] == "web_search"


def test_preference_skips_without_rejected_and_pairs_with_default() -> None:
    interaction = _tool_exchange()
    assert Formatter("preference").format_single(interaction) is None
    record = Formatter("preference", default_rejected="I don't know.").format_single(interaction)
    assert record["chosen"][0]["content"].strip()
    assert record["rejected"][0]["content"] == "I don't know."
    with pytest.raises(FormatValidationError):
        Formatter("preference", strict=True).format_single(interaction)


def test_alpaca_and_prompt_completion_tolerate_uneven_turns() -> None:
    interaction = _interaction(
        [
            CanonicalMessage(role="user", content="a"),
            CanonicalMessage(role="user", content="b"),
            CanonicalMessage(role="assistant", content="c"),
        ]
    )
    # Must not raise on more users than assistants.
    assert Formatter("alpaca").format_single(interaction)["output"] == "c"
    assert Formatter("prompt_completion").format_single(interaction)["completion"] == "c"


def test_canonical_format_is_lossless_with_provenance() -> None:
    interaction = _tool_exchange()
    interaction.model = "test-model"
    interaction.token_usage = {"input_tokens": 5}
    record = Formatter("canonical").format_single(interaction)
    assert record["model"] == "test-model"
    assert record["token_usage"] == {"input_tokens": 5}
    assert record["messages"][2]["tool_name"] == "web_search"


def test_include_metadata_is_off_by_default() -> None:
    interaction = _tool_exchange()
    interaction.model = "test-model"
    assert "_agentscribe" not in Formatter("openai_chat").format_single(interaction)
    enriched = Formatter("openai_chat", include_metadata=True).format_single(interaction)
    assert enriched["_agentscribe"]["model"] == "test-model"


def test_unknown_format_raises_with_supported_list() -> None:
    with pytest.raises(ValueError, match="Unsupported format"):
        Formatter("does_not_exist")


def test_format_messages_shares_path_with_facade() -> None:
    messages = _tool_exchange().to_dict()["messages"]
    record = format_messages(messages, "openai_chat")
    assert record["messages"][0]["role"] == "user"
