from __future__ import annotations

from types import SimpleNamespace

from agentscribe.adapters.crewai import (
    CrewAIAdapter,
    from_event,
    from_kickoff_output,
    from_llm_call_context,
    from_tool_call_context,
)


def test_from_llm_call_context_normalizes_messages_response_and_metadata() -> None:
    context = {
        "session_id": "s1",
        "run_id": "r1",
        "messages": [{"role": "user", "content": "Q"}],
        "response": "Final Answer: A",
        "agent": {"role": "Researcher", "goal": "Find facts", "llm": "model-a"},
        "task": {"description": "answer", "expected_output": "short"},
        "model": "model-a",
        "token_usage": {"total_tokens": 3},
        "iterations": 1,
    }

    interaction = from_llm_call_context(context, metadata={"case": "llm"})

    assert interaction.session_id == "s1"
    assert interaction.metadata["event"] == "llm_call"
    assert interaction.metadata["run_id"] == "r1"
    assert interaction.metadata["case"] == "llm"
    assert interaction.agent["role"] == "Researcher"
    assert interaction.metadata["task"]["description"] == "answer"
    assert interaction.model == "model-a"
    assert interaction.token_usage == {"total_tokens": 3}
    assert [message.content for message in interaction.messages] == ["Q", "Final Answer: A"]


def test_from_llm_call_context_accepts_minimal_empty_context_boundary() -> None:
    interaction = from_llm_call_context({})

    assert interaction.source_framework == "crewai"
    assert interaction.messages == []
    assert interaction.agent == {}


def test_from_tool_call_context_creates_call_and_response_messages() -> None:
    interaction = from_tool_call_context(
        {
            "crew_id": "crew-1",
            "execution_id": "run-1",
            "tool_name": "lookup",
            "tool_input": {"q": "x"},
            "tool_result": {"answer": "y"},
            "tool_call_id": "call-1",
            "agent": {"name": "agent"},
            "task": {"id": "task-1"},
        }
    )

    assert interaction.session_id == "crew-1"
    assert interaction.metadata["run_id"] == "run-1"
    assert [message.role for message in interaction.messages] == ["tool_call", "tool_response"]
    assert interaction.messages[0].tool_args["q"] == "x"
    assert interaction.messages[1].tool_result == '{"answer": "y"}'


def test_from_event_routes_tool_llm_and_unknown_events() -> None:
    tool = from_event({"source": "bus"}, {"event_type": "tool.finished", "tool_name": "t", "tool_result": "ok"})
    llm = from_event({"event_type": "agent.finished", "input": "Q", "output": "A"})
    unknown = from_event({"event_type": "misc", "prompt": "Q", "result": "A"})

    assert tool.metadata["event"] == "tool_call"
    assert llm.metadata["event"] == "llm_call"
    assert unknown.spans[0]["kind"] == "crewai.event"
    assert [message.content for message in unknown.messages] == ["Q", "A"]


def test_from_kickoff_output_records_final_output_usage_and_tasks() -> None:
    interaction = from_kickoff_output(
        {
            "raw": "finished",
            "token_usage": {"total_tokens": 4},
            "tasks_output": [SimpleNamespace(raw="task result")],
        },
        metadata={"case": "kickoff"},
    )

    assert interaction.messages[0].content == "finished"
    assert interaction.token_usage == {"total_tokens": 4}
    assert interaction.extra["tasks_output"] == [{"raw": "task result"}]
    assert interaction.metadata["case"] == "kickoff"


def test_crewai_adapter_helpers_merge_and_finalise_without_framework_import() -> None:
    adapter = CrewAIAdapter(flush_interval=99)
    context = {
        "crew": {"name": "crew"},
        "agent": {"role": "agent", "max_iter": 2},
        "task": {"description": "task"},
        "iterations": 2,
        "messages": [{"role": "user", "content": "Q"}],
        "response": "A",
    }

    assert adapter._resolve_session_id(context) == "crew:agent:task"
    assert adapter._is_final_iteration(context) is True

    adapter._on_after_tool({"tool_name": "lookup", "tool_result": "ok", **context})
    adapter._on_after_llm(context)

    assert len(adapter._buffer) == 1
    assert adapter._pending == {}


def test_crewai_adapter_final_iteration_ignores_invalid_iteration_values() -> None:
    adapter = CrewAIAdapter(flush_interval=99)

    assert adapter._is_final_iteration({"iterations": "bad", "agent": {"max_iter": 2}, "response": ""}) is False
    assert adapter._is_final_iteration({"response": "Final Answer: ok"}) is True
