from __future__ import annotations

from types import SimpleNamespace

import agentscribe.adapters.base as base_module
from agentscribe.adapters.base import BaseAdapter
from agentscribe.core.canonical import CanonicalInteraction


def _interaction(content: str = "ok") -> CanonicalInteraction:
    item = CanonicalInteraction(source_framework="unit", session_id="s1")
    item.add_message("assistant", content)
    return item


def test_finalise_buffers_until_flush_interval(monkeypatch) -> None:
    writes: list[tuple[str, list[dict], str]] = []
    monkeypatch.setattr(
        base_module,
        "write_jsonl",
        lambda output, records, mode="a": writes.append((output, list(records), mode)),
    )
    adapter = BaseAdapter(output="out.jsonl", flush_interval=2)
    adapter._pending["a"] = _interaction("first")
    adapter._pending["b"] = _interaction("second")

    adapter._finalise_and_flush("missing")
    adapter._finalise_and_flush("a")

    assert writes == []
    assert len(adapter._buffer) == 1

    adapter._finalise_and_flush("b")

    assert len(adapter._buffer) == 0
    assert writes == [("out.jsonl", [{"messages": [{"role": "assistant", "content": "first"}]}, {"messages": [{"role": "assistant", "content": "second"}]}], "a")]


def test_flush_interval_zero_writes_each_interaction(monkeypatch) -> None:
    writes: list[list[dict]] = []
    monkeypatch.setattr(base_module, "write_jsonl", lambda _output, records, mode="a": writes.append(list(records)))
    adapter = BaseAdapter(flush_interval=0)
    adapter._pending["a"] = _interaction()

    adapter._finalise_and_flush("a")

    assert len(writes) == 1
    assert adapter._buffer == []


def test_flush_restores_buffer_when_write_fails(monkeypatch) -> None:
    def fail_write(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(base_module, "write_jsonl", fail_write)
    adapter = BaseAdapter()
    buffered = _interaction()
    adapter._buffer.append(buffered)

    assert adapter.flush() == 0
    assert adapter._buffer == [buffered]


def test_context_manager_flushes_on_exit(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(BaseAdapter, "flush", lambda self: calls.append(self._output) or 1)

    with BaseAdapter(output="ctx.jsonl") as adapter:
        assert adapter._output == "ctx.jsonl"

    assert calls == ["ctx.jsonl"]
