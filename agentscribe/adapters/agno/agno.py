"""Agno adapter helpers.

Agno's current platform exposes runs, sessions, metrics, hooks, event streams,
AgentOS trace APIs, and OpenTelemetry/OpenInference tracing. These converters
accept the common run/session/trace shapes without importing Agno directly.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from agentscribe.core.canonical import CanonicalInteraction

from ..utils import (
	InteractionCollector,
	append_unique_message,
	as_list,
	build_metadata,
	compact_dict,
	get_value,
	interaction_from_messages,
	message_to_canonical,
	object_to_dict,
	serialize_object_list,
)


# ------------------------------------------------------------------
# Helpers (Agno-specific)
# ------------------------------------------------------------------
def _agent_metadata(agent: Any) -> dict[str, Any]:
	"""Build serialisable Agno agent metadata.

	Parameters
	----------
	agent : Any
		Agno agent object, team member object, workflow object, or mapping.

	Returns
	-------
	dict[str, Any]
		Compact metadata describing the agent, model, instructions, and tools.
	"""

	return build_metadata(
		agent,
		fields={
			"name": ("name",),
			"id": ("id", "agent_id"),
			"model": ("model",),
			"instructions": ("instructions", "system_message"),
			"tools": lambda value: serialize_object_list(get_value(value, "tools", default=[]) or []),
		},
	)


def _ids(record: Any) -> dict[str, str]:
	"""Resolve stable ids from an Agno run-like record.

	Parameters
	----------
	record : Any
		Agno run output, session run, event, or compatible mapping.

	Returns
	-------
	dict[str, str]
		Compact id mapping suitable for ``CanonicalInteraction`` construction.
	"""

	return build_metadata(
		record,
		fields={
			"session_id": ("session_id", "sessionId"),
		},
	)


def _provenance(record: Any) -> dict[str, str]:
	"""Resolve run and trace provenance from an Agno record.

	Parameters
	----------
	record : Any
		Agno run output, trace export, event, or compatible mapping.

	Returns
	-------
	dict[str, str]
		Compact provenance metadata.
	"""

	return build_metadata(
		record,
		fields={
			"run_id": ("run_id", "runId", "id"),
			"trace_id": ("trace_id", "traceId"),
		},
	)


def from_run_output(
	run_output: Any,
	*,
	agent: Any = None,
	metadata: Mapping[str, Any] | None = None,
) -> CanonicalInteraction:
	"""Normalize Agno Agent, Team, or Workflow run output.

	Parameters
	----------
	run_output : Any
		Agno run output object or compatible mapping.
	agent : Any
		Optional agent object used when metadata is not embedded in the output.
	metadata : Mapping[str, Any] | None
		Optional metadata to merge into the canonical interaction.

	Returns
	-------
	CanonicalInteraction
		Interaction containing message history or prompt/output data plus Agno metadata.
	"""

	messages = get_value(run_output, "messages", "chat_history", "history", default=None)
	if messages is not None:
		interaction = interaction_from_messages(
			messages,
			source_framework="agno",
			metadata={"source_shape": "run_output", **_provenance(run_output), **dict(metadata or {})},
			**_ids(run_output),
		)
	else:
		interaction = CanonicalInteraction(source_framework="agno", metadata={"source_shape": "run_output", **_provenance(run_output), **dict(metadata or {})}, **_ids(run_output))
		user_input = get_value(run_output, "input", "prompt", "message", default=None)
		output = get_value(run_output, "content", "output", "response", "raw", default=run_output)
		if user_input is not None:
			interaction.add_message("user", user_input)
		interaction.add_message("assistant", output)

	interaction.agent = _agent_metadata(agent or get_value(run_output, "agent", default=None))
	interaction.model = str(get_value(run_output, "model", default="") or "") or None
	metrics = object_to_dict(get_value(run_output, "metrics", "usage", "token_usage", default={}))
	if metrics:
		interaction.token_usage = metrics
	tools = get_value(run_output, "tools", "tool_calls", default=None)
	if tools is not None:
		interaction.tools = [object_to_dict(tool) or {"value": str(tool)} for tool in as_list(tools)]
	return interaction


def from_session(session: Any, *, metadata: Mapping[str, Any] | None = None) -> list[CanonicalInteraction]:
	"""Normalize an Agno session or AgentOS session export into interactions.

	Parameters
	----------
	session : Any
		Agno session export or compatible mapping containing runs.
	metadata : Mapping[str, Any] | None
		Optional metadata to merge into each canonical interaction.

	Returns
	-------
	list[CanonicalInteraction]
		One canonical interaction per run in the session.
	"""

	runs = get_value(session, "runs", "session_runs", "history", default=[])
	session_id = get_value(session, "session_id", "id", default=None)
	interactions = []
	for run in as_list(runs):
		interaction = from_run_output(run, metadata={"session_id": session_id, **dict(metadata or {})})
		if session_id and not interaction.session_id:
			interaction.session_id = str(session_id)
		interactions.append(interaction)
	return interactions


def from_trace(trace: Any, *, metadata: Mapping[str, Any] | None = None) -> CanonicalInteraction:
	"""Normalize an Agno trace export using the OpenTelemetry adapter.

	Parameters
	----------
	trace : Any
		Agno trace export, AgentOS trace record, or compatible mapping.
	metadata : Mapping[str, Any] | None
		Optional metadata to merge into the canonical interaction.

	Returns
	-------
	CanonicalInteraction
		Interaction inferred from trace spans and Agno trace metadata.
	"""

	from ..opentelemetry import from_spans

	spans = get_value(trace, "spans", default=None)
	if spans is None:
		spans = get_value(get_value(trace, "data", default={}), "spans", default=[trace])
	interaction = from_spans(as_list(spans), source_framework="agno", metadata={"source_shape": "trace", **dict(metadata or {})})
	trace_id = get_value(trace, "trace_id", "traceId", default=None)
	if trace_id is not None:
		interaction.trace_id = str(trace_id)
	return interaction


def from_event_stream(events: Iterable[Any], *, metadata: Mapping[str, Any] | None = None) -> CanonicalInteraction:
	"""Normalize Agno event-stream records from agents, teams, or workflows.

	Parameters
	----------
	events : Iterable[Any]
		Agno events from a live stream or exported event log.
	metadata : Mapping[str, Any] | None
		Optional metadata to merge into the canonical interaction.

	Returns
	-------
	CanonicalInteraction
		Interaction containing promoted event messages and raw event spans.
	"""

	interaction = CanonicalInteraction(source_framework="agno", metadata={"source_shape": "event_stream", **dict(metadata or {})})
	for event in events:
		event_type = str(get_value(event, "event", "type", "event_type", default=event.__class__.__name__))
		interaction.spans.append({"kind": "agno.event", "event_type": event_type, "event": object_to_dict(event) or str(event)})
		message = get_value(event, "message", "content", "delta", default=None)
		if message is not None:
			append_unique_message(interaction, message_to_canonical({"role": "assistant", "content": message, "type": event_type}))
	return interaction


class AgnoTraceCollector(InteractionCollector):
	"""Collector with hook-style methods for Agno post-run integration.

	Parameters
	----------
	format_name : str
		Output format used when flushing through the collector.
	output_path : str | None
		Optional path for collector flush output.

	Examples
	--------
	collector = AgnoTraceCollector(output_path="./agentscribe_data.jsonl")
	run_output = agent.run("Summarize this ticket")
	collector.record_run_output(run_output, agent=agent)
	collector.flush()
	"""

	def __init__(self, *, format_name: str = "openai_chat", output_path: str | None = None) -> None:
		"""Initialise the Agno trace collector.

		Parameters
		----------
		format_name : str
			Output format used when flushing through the collector.
		output_path : str | None
			Optional path for collector flush output.
		"""

		super().__init__(source_framework="agno", format_name=format_name, output_path=output_path)

	def record_run_output(self, run_output: Any, *, agent: Any = None) -> CanonicalInteraction:
		"""Record an Agno run output.

		Parameters
		----------
		run_output : Any
			Agno run output object or compatible mapping.
		agent : Any
			Optional agent object used for metadata.

		Returns
		-------
		CanonicalInteraction
			Recorded canonical interaction.
		"""

		return self.record(from_run_output(run_output, agent=agent))

	def post_hook(self, run_output: Any, *args: Any, **kwargs: Any) -> CanonicalInteraction:
		"""Record a run output from an Agno post-run hook.

		Parameters
		----------
		run_output : Any
			Agno run output object passed by the hook.
		*args : Any
			Additional hook positional arguments, ignored by AgentScribe.
		**kwargs : Any
			Additional hook keyword arguments. ``agent`` is used when present.

		Returns
		-------
		CanonicalInteraction
			Recorded canonical interaction.
		"""

		agent = kwargs.get("agent") if kwargs else None
		return self.record_run_output(run_output, agent=agent)


__all__ = [
	"AgnoTraceCollector",
	"from_event_stream",
	"from_run_output",
	"from_session",
	"from_trace",
]
