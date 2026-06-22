"""Collector helpers for adapter packages."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from agentscribe.core.canonical import CanonicalInteraction
from agentscribe.core.formatter import Formatter
from agentscribe.storage import write_jsonl


class AdapterError(RuntimeError):
    """Raised when an adapter cannot normalize or persist data."""


class InteractionCollector:
    """Small in-memory collector used by wrapper-style adapters."""

    def __init__(
        self,
        *,
        source_framework: str = "",
        format_name: str = "openai_chat",
        output_path: str | None = None,
    ) -> None:
        self.source_framework = source_framework
        self.format_name = format_name
        self.output_path = output_path
        self.interactions: list[CanonicalInteraction] = []

    def record(self, interaction: CanonicalInteraction) -> CanonicalInteraction:
        if not interaction.source_framework and self.source_framework:
            interaction.source_framework = self.source_framework
        self.interactions.append(interaction)
        return interaction

    def extend(self, interactions: Iterable[CanonicalInteraction]) -> None:
        for interaction in interactions:
            self.record(interaction)

    def format_records(self, format_name: str | None = None) -> list[dict[str, Any]]:
        formatter = Formatter(format_name or self.format_name)
        return [formatter.format_single(interaction) for interaction in self.interactions]

    def flush(self, output_path: str | None = None, *, append: bool = True, format_name: str | None = None) -> int:
        
        target = output_path or self.output_path
        if not target:
            raise AdapterError("flush requires an output_path")
        if not self.interactions:
            return 0

        # Snapshot what we're about to write so a concurrent record() (or a write
        # failure) can't desync the drain below.
        pending = self.interactions[:]
        formatter = Formatter(format_name or self.format_name)
        records = [formatter.format_single(interaction) for interaction in pending]
        result = write_jsonl(target, records, mode="a" if append else "w", format_name=format_name or self.format_name)

        # Drain only on append. In append mode each flush must write *new* records,
        # so we drop the ones we just persisted (this is what makes incremental
        # flushing idempotent and stops re-flush from duplicating). In overwrite
        # mode ("w") the file is the full snapshot every time, so we keep the
        # interactions and never drain. We only drain after a successful write.
        if append:
            del self.interactions[:len(pending)]
        return result.records_written


__all__ = ["AdapterError", "InteractionCollector"]