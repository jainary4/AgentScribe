from __future__ import annotations

from types import SimpleNamespace

import pytest

import agentscribe.adapters.utils.collector as collector_module
from agentscribe.adapters.utils.collector import AdapterError, InteractionCollector
from agentscribe.core.canonical import CanonicalInteraction


def _interaction(source_framework: str = "") -> CanonicalInteraction:
    interaction = CanonicalInteraction(source_framework=source_framework)
    interaction.add_message("assistant", "ok")
    return interaction


def test_record_applies_source_framework_fallback_and_extend_records_all() -> None:
    collector = InteractionCollector(source_framework="adapter")
    first = _interaction()
    second = _interaction("explicit")

    assert collector.record(first) is first
    collector.extend([second])

    assert [item.source_framework for item in collector.interactions] == ["adapter", "explicit"]


def test_format_records_uses_requested_format() -> None:
    collector = InteractionCollector()
    collector.record(_interaction())

    assert collector.format_records("prompt_completion") == [{"prompt": "", "completion": "ok"}]


def test_flush_requires_output_path() -> None:
    with pytest.raises(AdapterError, match="output_path"):
        InteractionCollector().flush()


def test_flush_writes_records_with_append_and_format(monkeypatch) -> None:
    calls: list[tuple[str, list[dict], str, str]] = []

    def fake_write(output, records, *, mode, format_name):
        calls.append((output, list(records), mode, format_name))
        return SimpleNamespace(records_written=len(calls[-1][1]))

    monkeypatch.setattr(collector_module, "write_jsonl", fake_write)
    collector = InteractionCollector(output_path="records.jsonl")
    collector.record(_interaction())

    assert collector.flush(append=True, format_name="openai_chat") == 1
    assert calls == [("records.jsonl", [{"messages": [{"role": "assistant", "content": "ok"}]}], "a", "openai_chat")]
