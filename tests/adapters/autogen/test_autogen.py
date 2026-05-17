from __future__ import annotations

from types import SimpleNamespace

from agentscribe.adapters.autogen import (
    from_chat_history,
    from_stream_events,
    from_task_result,
    messages_from_autogen_item,
)


def test_messages_from_tool_call_request_parse_json_arguments() -> None:
    messages = messages_from_autogen_item(
        {
            "type": "ToolCallRequestEvent",
            "source": "assistant",
            "content": [{"id": "call-1", "name": "search", "arguments": '{"q":"x"}'}],
        }
    )

    assert len(messages) == 1
    assert messages[0].role == "tool_call"
    assert messages[0].tool_name == "search"
    assert messages[0].tool_args["q"] == "x"
    assert messages[0].tool_calls[0]["id"] == "call-1"


def test_messages_from_tool_call_execution_marks_error_metadata() -> None:
    messages = messages_from_autogen_item(
        {
            "type": "ToolCallExecutionEvent",
            "content": [{"call_id": "call-1", "name": "search", "content": "nope", "is_error": True}],
        }
    )

    assert messages[0].role == "tool_response"
    assert messages[0].metadata["metadata"]["is_error"] is True


def test_messages_from_regular_item_sets_usage_and_stream_chunk_metadata() -> None:
    message = messages_from_autogen_item(
        {"type": "StreamingChunk", "source": "assistant", "content": "partial", "models_usage": {"total_tokens": 1}}
    )[0]

    assert message.role == "assistant"
    assert message.metadata["stream_chunk"] is True
    assert message.token_usage == {"total_tokens": 1}


def test_from_task_result_normalizes_messages_agent_usage_and_raw_payload() -> None:
    result = SimpleNamespace(
        messages=[{"source": "user", "content": "Q"}, {"source": "assistant", "content": "A"}],
        stop_reason="done",
        usage={"total_tokens": 2},
    )
    agent = SimpleNamespace(name="assistant", description="helper", system_message="system")

    interaction = from_task_result(result, agent=agent, metadata={"case": "task"}, preserve_raw=True)

    assert interaction.metadata["stop_reason"] == "done"
    assert interaction.metadata["case"] == "task"
    assert interaction.agent["name"] == "assistant"
    assert interaction.token_usage == {"total_tokens": 2}
    assert "raw_result" in interaction.extra
    assert [message.content for message in interaction.messages] == ["Q", "A"]


def test_from_chat_history_accepts_list_and_preserves_raw() -> None:
    interaction = from_chat_history({"chat_history": [{"role": "user", "content": "hello"}]}, preserve_raw=True)

    assert interaction.metadata["source_shape"] == "chat_history"
    assert interaction.extra["raw_chat_history"]["chat_history"][0]["content"] == "hello"


def test_from_stream_events_records_spans_and_final_result_messages() -> None:
    events = [
        {"type": "StreamingChunk", "content": "partial"},
        SimpleNamespace(messages=[{"source": "user", "content": "Q"}, {"source": "assistant", "content": "A"}]),
    ]

    interaction = from_stream_events(events)

    assert interaction.metadata["source_shape"] == "run_stream"
    assert len(interaction.spans) == 1
    assert [message.content for message in interaction.messages] == ["partial", "Q", "A"]


def test_from_stream_events_accepts_empty_iterable_boundary() -> None:
    interaction = from_stream_events([])

    assert interaction.messages == []
    assert interaction.spans == []
