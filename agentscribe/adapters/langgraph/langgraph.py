"""LangGraph adapter helpers.

LangGraph's most reliable public capture surface is the graph state and stream
API: compiled graphs expose invoke/stream-style methods, stream modes include
updates, values, messages, tasks, checkpoints, and debug, and chat state is
commonly stored under a messages key. This module keeps the integration
duck-typed so AgentScribe can parse exported states without requiring LangGraph
as an import-time dependency.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from agentscribe.core.canonical import CanonicalInteraction

from ..utils import (
	InteractionCollector,
	append_unique_message,
	as_list,
	compact_dict,
	get_value,
	interaction_from_messages,
	json_ready,
	message_to_canonical,
	object_to_dict,
)


# ------------------------------------------------------------------
# Helpers (LangGraph-specific)
# ------------------------------------------------------------------
def _thread_id_from_config(config: Any) -> str | None:
	"""Resolve a LangGraph thread id from invoke or stream config.

	Parameters
	----------
	config : Any
		LangGraph config object or mapping.

	Returns
	-------
	str | None
		Thread or checkpoint identifier when present.
	"""

	configurable = get_value(config, "configurable", default={})
	return get_value(configurable, "thread_id", "checkpoint_id", default=None)


def _graph_metadata(graph: Any) -> dict[str, Any]:
	"""Build serialisable LangGraph metadata.

	Parameters
	----------
	graph : Any
		Compiled graph, graph-like object, or ``None``.

	Returns
	-------
	dict[str, Any]
		Compact graph name and class metadata.
	"""

	if graph is None:
		return {}
	graph_name = get_value(graph, "name", "__name__", default=graph.__class__.__name__)
	return compact_dict(
		{
			"graph_name": graph_name,
			"graph_class": graph.__class__.__name__,
		}
	)


def _extract_messages(data: Any) -> list[Any]:
	"""Extract message history from common LangGraph state shapes.

	Parameters
	----------
	data : Any
		State snapshot, stream payload, or mapping containing nested messages.

	Returns
	-------
	list[Any]
		Messages found under direct or nested conversation keys.
	"""

	direct_messages = get_value(data, "messages", "chat_history", "conversation", default=None)
	if direct_messages is not None:
		return as_list(direct_messages)

	if isinstance(data, Mapping):
		messages: list[Any] = []
		for value in data.values():
			nested_messages = get_value(value, "messages", "chat_history", default=None)
			if nested_messages is not None:
				messages.extend(as_list(nested_messages))
		return messages
	return []


def from_state(
	state: Any,
	*,
	config: Any = None,
	graph: Any = None,
	session_id: str | None = None,
	metadata: Mapping[str, Any] | None = None,
) -> CanonicalInteraction:
	"""Normalize a LangGraph state snapshot into one canonical interaction.

	Parameters
	----------
	state : Any
		LangGraph state snapshot or compatible mapping.
	config : Any
		Optional invoke or stream config used to resolve the thread id.
	graph : Any
		Optional compiled graph used for metadata.
	session_id : str | None
		Explicit session id override.
	metadata : Mapping[str, Any] | None
		Optional metadata to merge into the canonical interaction.

	Returns
	-------
	CanonicalInteraction
		Interaction containing normalized messages, graph metadata, and extra state.
	"""

	state_mapping = object_to_dict(state)
	thread_id = session_id or _thread_id_from_config(config) or get_value(state, "thread_id", "session_id", default=None)
	interaction = interaction_from_messages(
		_extract_messages(state),
		source_framework="langgraph",
		session_id=str(thread_id) if thread_id is not None else None,
		thread_id=str(thread_id) if thread_id is not None else None,
		metadata={**dict(metadata or {}), **_graph_metadata(graph)},
	)
	interaction.instantiation = compact_dict(
		{
			"config": object_to_dict(config),
			"graph": _graph_metadata(graph),
		}
	)
	extra_state = {key: value for key, value in state_mapping.items() if key not in {"messages", "chat_history"}}
	if extra_state:
		interaction.extra["state"] = json_ready(extra_state)
	return interaction


def normalize_stream_event(event: Any) -> dict[str, Any]:
	"""Normalize LangGraph stream outputs across tuple and StreamPart forms.

	Parameters
	----------
	event : Any
		Stream tuple, mapping, StreamPart-like object, or raw event payload.

	Returns
	-------
	dict[str, Any]
		Compact mapping containing stream mode, namespace, and data when available.
	"""

	if isinstance(event, Mapping):
		return compact_dict(
			{
				"mode": get_value(event, "type", "mode", "stream_mode", default=None),
				"namespace": get_value(event, "ns", "namespace", default=None),
				"data": get_value(event, "data", "value", "event", default=event),
			}
		)
	if isinstance(event, tuple):
		if len(event) == 3:
			mode, namespace, data = event
			return compact_dict({"mode": mode, "namespace": namespace, "data": data})
		if len(event) == 2:
			first_item, second_item = event
			if isinstance(first_item, str):
				return compact_dict({"mode": first_item, "data": second_item})
			return {"mode": "messages", "data": event}
	return {"mode": "unknown", "data": event}


def from_stream_events(
	events: Iterable[Any],
	*,
	config: Any = None,
	graph: Any = None,
	include_message_chunks: bool = False,
	metadata: Mapping[str, Any] | None = None,
) -> CanonicalInteraction:
	"""Build an interaction from LangGraph stream events.

	Parameters
	----------
	events : Iterable[Any]
		LangGraph stream events from ``graph.stream()`` or an exported stream log.
	config : Any
		Optional stream config used to resolve the thread id.
	graph : Any
		Optional compiled graph used for metadata.
	include_message_chunks : bool
		Whether token/message chunks from ``messages`` mode should become dataset messages.
	metadata : Mapping[str, Any] | None
		Optional metadata to merge into the canonical interaction.

	Returns
	-------
	CanonicalInteraction
		Interaction containing promoted messages plus raw stream events as spans.

	Notes
	-----
	Final state messages from updates/values are promoted into canonical
	messages. Token chunks from messages mode are preserved as spans by default.
	"""

	thread_id = _thread_id_from_config(config)
	interaction = CanonicalInteraction(
		source_framework="langgraph",
		session_id=thread_id,
		metadata={**dict(metadata or {}), **_graph_metadata(graph)},
	)
	interaction.thread_id = thread_id
	interaction.instantiation = compact_dict({"config": object_to_dict(config), "graph": _graph_metadata(graph)})

	for event in events:
		normalized = normalize_stream_event(event)
		interaction.spans.append({"kind": "langgraph.stream", **json_ready(normalized)})
		mode = str(normalized.get("mode", ""))
		data = normalized.get("data")
		if mode in {"updates", "values", "debug", "tasks"}:
			for message in _extract_messages(data):
				append_unique_message(interaction, message_to_canonical(message))
		elif mode == "messages" and include_message_chunks:
			chunk = data[0] if isinstance(data, (tuple, list)) and data else data
			append_unique_message(
				interaction,
				message_to_canonical(chunk, metadata={"stream_mode": "messages", "stream_chunk": True}),
			)
	return interaction


class LangGraphRecorder:
	"""Thin wrapper around a compiled graph that records invoke/stream results.

	Parameters
	----------
	graph : Any
		Compiled LangGraph graph or compatible object with ``invoke`` and ``stream``.
	collector : InteractionCollector | None
		Optional collector used to store interactions.

	Examples
	--------
	wrapped = LangGraphRecorder(graph)
	state = wrapped.invoke({"messages": [{"role": "user", "content": "Hi"}]})
	wrapped.collector.flush()
	"""

	def __init__(self, graph: Any, *, collector: InteractionCollector | None = None) -> None:
		"""Initialise the recorder.

		Parameters
		----------
		graph : Any
			Compiled LangGraph graph or compatible object.
		collector : InteractionCollector | None
			Optional collector used to store interactions.
		"""

		self.graph = graph
		self.collector = collector or InteractionCollector(source_framework="langgraph")

	def invoke(self, *args: Any, **kwargs: Any) -> Any:
		"""Run ``graph.invoke`` and record the resulting state.

		Parameters
		----------
		*args : Any
			Positional arguments passed to the wrapped graph.
		**kwargs : Any
			Keyword arguments passed to the wrapped graph.

		Returns
		-------
		Any
			The original graph invoke result.
		"""

		result = self.graph.invoke(*args, **kwargs)
		self.collector.record(from_state(result, config=kwargs.get("config"), graph=self.graph))
		return result

	def stream(self, *args: Any, **kwargs: Any) -> list[Any]:
		"""Run ``graph.stream`` and record the collected stream events.

		Parameters
		----------
		*args : Any
			Positional arguments passed to the wrapped graph.
		**kwargs : Any
			Keyword arguments passed to the wrapped graph.

		Returns
		-------
		list[Any]
			The collected stream events.
		"""

		events = list(self.graph.stream(*args, **kwargs))
		self.collector.record(from_stream_events(events, config=kwargs.get("config"), graph=self.graph))
		return events


def wrap_graph(graph: Any, *, collector: InteractionCollector | None = None) -> LangGraphRecorder:
	"""Return a recording wrapper without mutating the original graph.

	Parameters
	----------
	graph : Any
		Compiled LangGraph graph or compatible object.
	collector : InteractionCollector | None
		Optional collector used by the wrapper.

	Returns
	-------
	LangGraphRecorder
		Recording wrapper around the supplied graph.
	"""

	return LangGraphRecorder(graph, collector=collector)


__all__ = [
	"LangGraphRecorder",
	"from_state",
	"from_stream_events",
	"normalize_stream_event",
	"wrap_graph",
]
