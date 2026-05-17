from __future__ import annotations

import pytest
import click

from agentscribe.adapters.utils.registry import ADAPTER_RECORD_LOADERS, adapter_record_to_interactions


def test_registry_contains_all_supported_adapter_sources_and_aliases() -> None:
    assert set(ADAPTER_RECORD_LOADERS) == {
        "ag2",
        "agentops",
        "agno",
        "atomic",
        "atomic_agents",
        "autogen",
        "crewai",
        "langgraph",
        "mcp",
        "mlflow",
        "opentelemetry",
        "openinference",
    }


@pytest.mark.parametrize(
    ("source", "record"),
    [
        ("langgraph", {"state": {"messages": [{"role": "user", "content": "lg"}]}}),
        ("crewai", {"messages": [{"role": "user", "content": "crew"}], "response": "done"}),
        ("agno", {"content": "agno"}),
        ("autogen", {"messages": [{"source": "user", "content": "auto"}]}),
        ("ag2", {"chat_history": [{"role": "user", "content": "ag2"}]}),
        ("atomic", {"response": "atomic", "input": "prompt"}),
        ("atomic_agents", {"event_type": "event", "output": "atomic-event"}),
        ("agentops", {"spans": [{"input.value": "agentops"}]}),
        ("mlflow", {"spans": [{"input.value": "mlflow"}]}),
        ("opentelemetry", {"spans": [{"input.value": "otel"}]}),
        ("openinference", {"spans": [{"input.value": "oi"}]}),
        ("mcp", {"request": {"id": 1, "method": "tools/list"}, "response": {"id": 1, "result": {"tools": []}}}),
    ],
)
def test_adapter_record_to_interactions_dispatches_each_source(source: str, record: dict) -> None:
    interactions = adapter_record_to_interactions(record, source=source)

    assert len(interactions) >= 1
    assert all(interaction.source_framework for interaction in interactions)


def test_adapter_record_to_interactions_raises_for_unknown_source() -> None:
    with pytest.raises(click.ClickException, match="No adapter parser"):
        adapter_record_to_interactions({}, source="missing")


def test_mcp_loader_raises_for_malformed_record() -> None:
    with pytest.raises(click.ClickException, match="MCP records"):
        adapter_record_to_interactions({}, source="mcp")
