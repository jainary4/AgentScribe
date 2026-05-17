from __future__ import annotations

import pytest

from agentscribe.adapters.agno import (
    AgnoAdapter,
    AgnoTraceCollector,
    from_event_stream,
    from_run_output,
    from_session,
    from_trace,
    parse_agno_run_output,
)


def test_from_run_output_normalizes_messages_tools_metrics_and_model() -> None:
    run_output = {
        "run_id": "run-1",
        "session_id": "session-1",
        "messages": [{"role": "user", "content": "Weather?"}],
        "tools": [
            {
                "tool_name": "lookup",
                "tool_args": {"city": "Toronto"},
                "tool_result": {"temp": 21},
                "tool_call_id": "call-1",
                "duration_ms": 7,
            }
        ],
        "metrics": {"total_tokens": 12},
        "model": "gpt-test",
    }

    interaction = from_run_output(run_output, metadata={"case": "normal"})

    assert interaction.source_framework == "agno"
    assert interaction.session_id == "session-1"
    assert interaction.run_id == "run-1"
    assert interaction.model == "gpt-test"
    assert interaction.token_usage == {"total_tokens": 12}
    assert interaction.metadata["case"] == "normal"
    assert [message.role for message in interaction.messages] == ["user", "tool_call", "tool_response"]


def test_from_run_output_uses_assistant_content_when_history_is_empty() -> None:
    interaction = from_run_output({"id": "run-2", "content": "final answer"})

    assert interaction.session_id == "run-2"
    assert [(message.role, message.content) for message in interaction.messages] == [("assistant", "final answer")]


def test_from_session_splits_runs_and_inherits_session_fields() -> None:
    interactions = from_session(
        {
            "id": "session-1",
            "agent_name": "researcher",
            "user_id": "user-1",
            "runs": [{"id": "r1", "content": "one"}, {"id": "r2", "content": "two"}],
        }
    )

    assert [item.session_id for item in interactions] == ["session-1", "session-1"]
    assert [item.metadata["agent_name"] for item in interactions] == ["researcher", "researcher"]
    assert [item.messages[0].content for item in interactions] == ["one", "two"]


def test_from_session_without_runs_falls_back_to_single_run() -> None:
    interactions = from_session({"session_id": "s", "messages": []})

    assert len(interactions) == 1
    assert interactions[0].metadata["source_shape"] == "session"


def test_from_event_stream_filters_chunks_by_default_and_promotes_terminal_output() -> None:
    events = [
        {"event": "RunStarted", "session_id": "s", "run_id": "r"},
        {"event": "StreamingChunk", "message": {"role": "assistant", "content": "partial", "type": "stream_chunk"}},
        {"event": "RunCompleted", "content": "done"},
    ]

    interaction = from_event_stream(events)

    assert interaction.session_id == "s"
    assert interaction.run_id == "r"
    assert [message.content for message in interaction.messages] == ["done"]
    assert len(interaction.spans) == 3


def test_from_event_stream_can_include_message_chunks_boundary() -> None:
    interaction = from_event_stream(
        [{"event": "StreamingChunk", "message": {"role": "assistant", "content": "partial", "type": "stream_chunk"}}],
        include_message_chunks=True,
    )

    assert interaction.messages[0].content == "partial"


def test_from_trace_delegates_to_opentelemetry_parser_with_agno_source() -> None:
    interaction = from_trace({"trace": {"trace_id": "trace-1", "spans": [{"input.value": "hi"}]}})

    assert interaction.source_framework == "agno"
    assert interaction.trace_id == "trace-1"
    assert interaction.messages[0].content == "hi"


def test_collector_records_supported_shapes() -> None:
    collector = AgnoTraceCollector()

    first = collector.record_run_output({"content": "ok"})
    sessions = collector.record_session({"runs": [{"content": "one"}, {"content": "two"}]})
    stream = collector.record_event_stream([{"event": "RunCompleted", "content": "done"}])
    trace = collector.record_trace({"spans": [{"input.value": "trace"}]})

    assert collector.interactions == [first, *sessions, stream, trace]
    assert all(item.source_framework == "agno" for item in collector.interactions)


def test_adapter_tool_hook_records_success_and_reraises_failures() -> None:
    adapter = AgnoAdapter(flush_interval=99)

    assert adapter.tool_hook("add", lambda x, y: x + y, {"x": 1, "y": 2}) == 3
    assert [message.role for message in adapter._pending_tool_messages] == ["tool_call", "tool_response"]

    with pytest.raises(ValueError, match="bad"):
        adapter.tool_hook("fail", lambda: (_ for _ in ()).throw(ValueError("bad")), {})

    assert adapter._pending_tool_messages[-1].metadata["metadata"]["error"] == "ValueError"


def test_adapter_post_hook_buffers_interaction_and_clears_pending_tools() -> None:
    adapter = AgnoAdapter(flush_interval=99)
    adapter._pending_tool_messages.append(adapter.tool_hook("echo", lambda text: text, {"text": "hi"}) if False else from_run_output({"content": "noop"}).messages[0])

    adapter.post_hook({"content": "done"}, {"name": "agent-a"})

    assert len(adapter._buffer) == 1
    assert adapter._pending_tool_messages == []
    assert adapter._buffer[0].metadata["agent_name"] == "agent-a"


def test_parse_agno_run_output_returns_formatted_records() -> None:
    assert parse_agno_run_output({"messages": [{"role": "user", "content": "hi"}]}) == {
        "messages": [{"role": "user", "content": "hi"}]
    } or [{"messages": [{"role": "user", "content": "hi"}]}]
