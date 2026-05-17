from __future__ import annotations

from agentscribe.adapters.agno import AgnoTraceCollector
from agentscribe.adapters.utils import InteractionCollector
from agentscribe.core.canonical import CanonicalInteraction

from .conftest import read_jsonl_file


def test_interaction_collector_flush_formats_and_writes_local_jsonl(tmp_path) -> None:
    output_path = tmp_path / "collector.jsonl"
    interaction = CanonicalInteraction(source_framework="")
    interaction.add_message("user", "Q")
    interaction.add_message("assistant", "A")
    collector = InteractionCollector(source_framework="custom", output_path=str(output_path))

    collector.record(interaction)

    assert collector.flush(format_name="prompt_completion") == 1
    assert interaction.source_framework == "custom"
    assert read_jsonl_file(output_path) == [{"prompt": "Q", "completion": "A"}]


def test_framework_collector_converts_records_and_flushes_through_shared_collector(tmp_path) -> None:
    output_path = tmp_path / "agno.jsonl"
    collector = AgnoTraceCollector(format_name="openai_chat", output_path=str(output_path))

    collector.record_run_output(
        {
            "messages": [{"role": "user", "content": "Q"}],
            "content": "A",
            "run_id": "run-1",
        }
    )

    assert collector.flush() == 1
    assert read_jsonl_file(output_path) == [{"messages": [{"role": "user", "content": "Q"}]}]
