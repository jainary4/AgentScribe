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


def test_from_chat_history_corrects_ag2_inverted_roles() -> None:
    # AG2 stores chat_history from the initiator's perspective: the human's turn
    # is recorded role="assistant" and the agent's reply role="user". The adapter
    # must resolve user/assistant by `name`, not the inverted `role`. (Format
    # validation can't catch this — an inverted transcript is still valid-shaped.)
    history = [
        {"role": "system", "content": "You are helpful."},
        {"content": "Translate 'hi'.", "role": "assistant", "name": "user"},   # actually the human
        {"content": "Bonjour", "role": "user", "name": "plain"},               # actually the agent
    ]
    interaction = from_chat_history(history)

    assert [m.role for m in interaction.messages] == ["system", "user", "assistant"]
    assert interaction.messages[1].content == "Translate 'hi'."
    assert interaction.messages[2].content == "Bonjour"


def test_from_chat_history_parses_nested_tool_calls_with_clean_args() -> None:
    # Real AG2 tool shape: name/arguments are nested under `function`, the tool
    # call carries a real id, and there is no internal bookkeeping in the args.
    history = [
        {"role": "system", "content": "Use multiply."},
        {"content": "6 times 7.", "role": "assistant", "name": "user"},
        {"tool_calls": [{"id": "call_x", "type": "function",
                         "function": {"name": "multiply", "arguments": '{"a": 6, "b": 7}'}}],
         "content": None, "role": "assistant"},
        {"content": "42", "tool_responses": [{"tool_call_id": "call_x", "role": "tool", "content": "42"}],
         "role": "tool", "name": "user"},
    ]
    interaction = from_chat_history(history)

    assert [m.role for m in interaction.messages] == ["system", "user", "tool_call", "tool_response"]
    call = next(m for m in interaction.messages if m.role == "tool_call")
    response = next(m for m in interaction.messages if m.role == "tool_response")
    assert call.tool_name == "multiply" and call.tool_call_id == "call_x"
    assert call.tool_args == {"a": 6, "b": 7}          # no internal _agentscribe leak
    assert response.tool_call_id == "call_x"           # response links back to the call


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
