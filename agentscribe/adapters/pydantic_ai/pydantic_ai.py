
"""Pydantic AI adapter for AgentScribe.
 
Pydantic AI executes tools itself and stores the full structured transcript on the
run result: ``result.all_messages()`` is a list of ``ModelRequest`` / ``ModelResponse``
objects whose ``parts`` carry the user prompt, the assistant text, and — crucially —
``ToolCallPart`` (tool_name, args, tool_call_id) and ``ToolReturnPart`` (tool_name,
content, tool_call_id). So, unlike Atomic Agents, we get structured tool calls for free
by walking the transcript; no tool hook is required.
 
Usage (capture one run):
    from agentscribe.adapters.pydantic_ai import from_run
    from agentscribe.core.formatter import Formatter
    from agentscribe.storage import write_jsonl
 
    result = agent.run_sync("What is 21 * 2?")
    write_jsonl("data.jsonl", [Formatter("openai_chat").format_single(from_run(result))])
 
Usage (batch / auto):
    from agentscribe.adapters.pydantic_ai import PydanticAITraceCollector, PydanticAIAdapter
 
    collector = PydanticAITraceCollector(format_name="openai_chat", output_path="data.jsonl")
    collector.record_run(agent.run_sync("..."))
    collector.flush()
 
    adapter = PydanticAIAdapter(format="openai_chat", output="data.jsonl")
    adapter.capture(agent.run_sync("..."))   # buffered; auto-flushed at exit
"""
 
from __future__ import annotations
 
import logging
from collections.abc import Mapping
from typing import Any
 
from agentscribe.adapters.base import BaseAdapter
from agentscribe.core.canonical import CanonicalInteraction, CanonicalMessage
 
from ..utils import (
    InteractionCollector,
    as_list,
    coerce_text,
    compact_dict,
    get_value,
    json_ready,
    parse_jsonish,
)
 
_logger = logging.getLogger("agentscribe.pydantic_ai")
 
 
# --------------------------------------------------------------------------- #
# Part -> canonical message translation
# --------------------------------------------------------------------------- #
def _clean_args(args: Any) -> dict[str, Any]:
    """Return tool-call arguments as a clean JSON-ready dict.
 
    Pydantic AI's ``ToolCallPart.args`` is usually already a dict, but some models
    deliver it as a JSON string; ``parse_jsonish`` handles both.
    """
 
    parsed = parse_jsonish(args)
    if isinstance(parsed, Mapping):
        return {str(key): json_ready(value) for key, value in parsed.items()}
    return {} if parsed is None else {"value": json_ready(parsed)}
 
 
def _part_to_message(part: Any) -> CanonicalMessage | None:
    """Translate one Pydantic AI message ``part`` into a canonical message.
 
    Keyed on the ``part_kind`` discriminator so we never import pydantic_ai types
    (the dependency stays optional). Returns ``None`` for parts we intentionally
    drop (e.g. ``thinking``).
    """
 
    kind = str(get_value(part, "part_kind", default="") or "")
    content = get_value(part, "content", default=None)
    tool_name = get_value(part, "tool_name", default=None)
    call_id = get_value(part, "tool_call_id", default=None)
    call_id = str(call_id) if call_id is not None else None
 
    if kind == "system-prompt":
        return CanonicalMessage(role="system", content=coerce_text(content))
    if kind == "user-prompt":
        return CanonicalMessage(role="user", content=coerce_text(content))
    if kind == "text":
        return CanonicalMessage(role="assistant", content=coerce_text(content))
    if kind == "tool-call":
        return CanonicalMessage(
            role="tool_call", content="",
            tool_name=str(tool_name) if tool_name else None,
            tool_args=_clean_args(get_value(part, "args", default={})),
            tool_call_id=call_id,
        )
    if kind == "tool-return":
        return CanonicalMessage(
            role="tool_response", content=coerce_text(content),
            tool_name=str(tool_name) if tool_name else None,
            tool_result=coerce_text(content) if content is not None else None,
            tool_call_id=call_id,
        )
    if kind == "retry-prompt":
        # A validation/tool error fed back to the model. If it carries a tool_name
        # it's a tool retry (tool_response); otherwise it's a plain user-side retry.
        return CanonicalMessage(
            role="tool_response" if tool_name else "user",
            content=coerce_text(content),
            tool_name=str(tool_name) if tool_name else None,
            tool_result=coerce_text(content) if tool_name else None,
            tool_call_id=call_id,
        )
    return None  # thinking / builtin-tool / unknown -> skip
 
 
def _model_name(messages: Any) -> str | None:
    """Resolve the model name from the most recent ``ModelResponse``."""
 
    for message in reversed(as_list(messages)):
        name = get_value(message, "model_name", default=None)
        if name:
            return str(name)
    return None
 
 
def _usage_dict(result: Any) -> dict[str, Any]:
    """Build a compact token-usage dict from ``result.usage`` (property or method)."""
 
    usage = get_value(result, "usage", default=None)
    # Newer pydantic-ai exposes ``usage`` as a property returning a usage object;
    # older versions exposed ``usage()`` as a method. Only call it when what we got
    # isn't already a usage object (no token fields) but is callable.
    if usage is not None and not hasattr(usage, "input_tokens") and callable(usage):
        try:
            usage = usage()
        except Exception:
            usage = None
    if usage is None:
        return {}
    fields = ("input_tokens", "output_tokens", "total_tokens", "request_tokens",
              "cache_read_tokens", "cache_write_tokens", "requests")
    return compact_dict({field: get_value(usage, field, default=None) for field in fields})
 
 
# --------------------------------------------------------------------------- #
# Converters (public API)
# --------------------------------------------------------------------------- #
def from_messages(messages: Any, *, metadata: Mapping[str, Any] | None = None) -> CanonicalInteraction:
    """Normalize a list of Pydantic AI ``ModelMessage`` objects into one interaction."""
 
    interaction = CanonicalInteraction(source_framework="pydantic_ai", metadata=dict(metadata or {}))
    for message in as_list(messages):
        for part in get_value(message, "parts", default=[]) or []:
            canonical = _part_to_message(part)
            if canonical is not None:
                interaction.messages.append(canonical)
    model = _model_name(messages)
    if model:
        interaction.model = model
    return interaction
 
 
def from_run(result: Any, *, metadata: Mapping[str, Any] | None = None) -> CanonicalInteraction:
    """Normalize a single ``AgentRunResult`` (one ``agent.run`` / ``run_sync``).
 
    Uses ``new_messages()`` so a run made inside a longer conversation captures only
    that run's turns, not the whole history. Run provenance (run_id, conversation_id),
    model, token usage, and the final ``.output`` are attached as metadata.
    """
 
    messages = result.new_messages() if hasattr(result, "new_messages") else as_list(result)
    interaction = from_messages(messages, metadata={"source_shape": "run", **dict(metadata or {})})
 
    run_id = get_value(result, "run_id", default=None)
    conversation_id = get_value(result, "conversation_id", default=None)
    if run_id is not None:
        interaction.run_id = str(run_id)
    if conversation_id is not None:
        interaction.session_id = str(conversation_id)
 
    usage = _usage_dict(result)
    if usage:
        interaction.token_usage = usage
 
    output = get_value(result, "output", default=None)
    if output is not None:
        interaction.extra["output"] = json_ready(output)   # structured output preserved
    return interaction
 
 
def from_session(result_or_messages: Any, *, metadata: Mapping[str, Any] | None = None) -> CanonicalInteraction:
    """Normalize a whole conversation into one interaction.
 
    Accepts either an ``AgentRunResult`` (whose ``all_messages()`` spans every run in
    the conversation) or a message list reloaded from storage (e.g. via
    ``ModelMessagesTypeAdapter``).
    """
 
    if hasattr(result_or_messages, "all_messages"):
        messages = result_or_messages.all_messages()
    else:
        messages = as_list(result_or_messages)
    interaction = from_messages(messages, metadata={"source_shape": "session", **dict(metadata or {})})
 
    conversation_id = get_value(result_or_messages, "conversation_id", default=None)
    if conversation_id is not None:
        interaction.session_id = str(conversation_id)
    return interaction
 
 
def from_trace(trace: Any, *, metadata: Mapping[str, Any] | None = None) -> CanonicalInteraction:
    """Normalize a Pydantic AI OpenTelemetry/Logfire trace into one interaction.
 
    Reuses AgentScribe's shared OpenTelemetry normalizer; Pydantic AI emits standard
    ``gen_ai.*`` spans, so no Pydantic-AI-specific parsing is required here.
    """
 
    from ..opentelemetry import from_trace as from_otel_trace
 
    payload = get_value(trace, "trace", default=trace)
    interaction = from_otel_trace(
        payload, source_framework="pydantic_ai",
        metadata={"source_shape": "trace", **dict(metadata or {})},
    )
    trace_id = get_value(payload, "trace_id", "traceId", "id", default=None)
    if trace_id is not None:
        interaction.trace_id = str(trace_id)
    return interaction
 
 
# --------------------------------------------------------------------------- #
# Collector (batch) + adapter (live)
# --------------------------------------------------------------------------- #
class PydanticAITraceCollector(InteractionCollector):
    """In-memory batch collector for Pydantic AI runs, sessions, and traces."""
 
    def __init__(self, *, format_name: str = "openai_chat", output_path: str | None = None) -> None:
        super().__init__(source_framework="pydantic_ai", format_name=format_name, output_path=output_path)
 
    def record_run(self, result: Any) -> CanonicalInteraction:
        return self.record(from_run(result))
 
    def record_session(self, result_or_messages: Any) -> CanonicalInteraction:
        return self.record(from_session(result_or_messages))
 
    def record_trace(self, trace: Any) -> CanonicalInteraction:
        return self.record(from_trace(trace))
 
 
class PydanticAIAdapter(BaseAdapter):
    """Live capture: call ``capture(result)`` after each run; buffered + auto-flushed.
 
    Pydantic AI has no global post-run hook, but its transcript already contains the
    structured tool calls, so a single ``capture(result)`` after ``agent.run(...)`` is
    all that's needed — there is nothing to wire per tool.
    """
 
    def capture(self, result: Any) -> None:
        """Capture one run result (uses ``new_messages()``)."""
 
        try:
            self._finalise_one(from_run(result))
        except Exception as exc:
            _logger.error("PydanticAIAdapter.capture failed: %s", exc)
 
    def capture_session(self, result_or_messages: Any) -> None:
        """Capture a whole conversation (uses ``all_messages()`` / a message list)."""
 
        try:
            self._finalise_one(from_session(result_or_messages))
        except Exception as exc:
            _logger.error("PydanticAIAdapter.capture_session failed: %s", exc)
 
    def _finalise_one(self, interaction: CanonicalInteraction) -> None:
        with self._lock:
            self._buffer.append(interaction)
            if self._flush_interval <= 0 or len(self._buffer) >= self._flush_interval:
                self._flush_buffer()
 
 
__all__ = [
    "PydanticAIAdapter",
    "PydanticAITraceCollector",
    "from_messages",
    "from_run",
    "from_session",
    "from_trace",
]