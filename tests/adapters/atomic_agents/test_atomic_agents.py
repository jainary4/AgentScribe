from __future__ import annotations

from types import SimpleNamespace

from agentscribe.adapters.atomic_agents import (
    AtomicAgentsTraceCollector,
    from_agent_response,
    from_agent_run,
    from_chat_history,
    from_log_event,
)


def test_from_chat_history_normalizes_history_and_agent_metadata() -> None:
    agent = SimpleNamespace(
        config=SimpleNamespace(
            system_prompt="Use facts",
            model="model-a",
            input_schema=str,
            output_schema=dict,
            tools=[{"name": "search"}],
            context_providers=[],
        )
    )

    interaction = from_chat_history({"messages": [{"role": "user", "content": "hi"}]}, agent=agent)

    assert interaction.source_framework == "atomic_agents"
    assert interaction.metadata["source_shape"] == "chat_history"
    assert interaction.messages[0].content == "hi"
    assert interaction.agent["system_prompt"] == "Use facts"
    assert interaction.agent["input_schema"] == "str"


def test_from_chat_history_accepts_empty_history_boundary() -> None:
    interaction = from_chat_history([])

    assert interaction.messages == []
    assert interaction.agent == {}


def test_from_agent_response_adds_prompt_and_structured_output() -> None:
    response = SimpleNamespace(answer="42", confidence=1)

    interaction = from_agent_response(response, prompt={"question": "life"})

    assert [message.role for message in interaction.messages] == ["user", "assistant"]
    assert interaction.extra["structured_output"] == {"answer": "42", "confidence": 1}
    assert interaction.instantiation == {}


def test_from_agent_response_uses_history_without_duplicating_existing_response() -> None:
    interaction = from_agent_response(
        "A",
        history={"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]},
    )

    assert [(message.role, message.content) for message in interaction.messages] == [("user", "Q"), ("assistant", "A")]
    assert interaction.metadata["source_shape"] == "agent_response"


def test_from_agent_run_delegates_input_output_pair() -> None:
    interaction = from_agent_run("prompt", "answer")

    assert [message.content for message in interaction.messages] == ["prompt", "answer"]


def test_from_log_event_records_span_and_optional_io() -> None:
    interaction = from_log_event({"event_type": "agent.run", "input": "Q", "output": "A"}, metadata={"case": "log"})

    assert interaction.metadata["event_type"] == "agent.run"
    assert interaction.metadata["case"] == "log"
    assert [message.content for message in interaction.messages] == ["Q", "A"]
    assert interaction.spans[0]["kind"] == "atomic_agents.event"


def test_collector_records_responses_and_log_events() -> None:
    collector = AtomicAgentsTraceCollector()

    response = collector.record_response("A", prompt="Q")
    event = collector.on_log_event({"output": "done"})

    assert collector.interactions == [response, event]
    assert all(item.source_framework == "atomic_agents" for item in collector.interactions)
