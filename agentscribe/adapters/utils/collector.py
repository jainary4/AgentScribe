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

    def flush(self, output_path: str | None = None, *, append: bool = False, format_name: str | None = None) -> int:
        target = output_path or self.output_path
        if not target:
            raise AdapterError("flush requires an output_path")
        records = self.format_records(format_name=format_name)
        result = write_jsonl(target, records, mode="a" if append else "w", format_name=format_name or self.format_name)
        return result.records_written


__all__ = ["AdapterError", "InteractionCollector"]