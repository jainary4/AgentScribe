"""AutoGen and AG2 adapter helpers.

Role handling note
------------------
AG2's legacy ``ConversableAgent`` stores a conversation from the *initiator's*
perspective: the messages it sends are recorded as ``role="assistant"`` and the
replies it receives as ``role="user"``. So in a ``ChatResult.chat_history`` (or a
``RunResponse.messages``) the ``role`` field is **inverted**, while the ``name``
field is always the true author. We therefore resolve user/assistant turns by
``name`` (using the first plain turn — the initiator's opening message — to
identify the human side), and only trust ``role`` for ``system``/``tool`` turns.

The newer AgentChat shape instead carries a ``source`` field, which is *not*
inverted and is used directly.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from agentscribe.core.canonical import CanonicalInteraction, CanonicalMessage

from ..utils import (
	append_unique_message,
	as_list,
	coerce_text,
	compact_dict,
	function_call_to_tool_call,
	get_nested,
	get_value,
	json_ready,
	message_to_canonical,
	normalize_role,
	object_to_dict,
	parse_jsonish,
	tool_call_message,
	tool_response_message,
)


def _autogen_type(item: Any) -> str:
	return str(get_value(item, "type", "event_type", default=item.__class__.__name__))


def _usage(item: Any) -> dict[str, Any]:
	return object_to_dict(get_value(item, "models_usage", "usage", "token_usage", default={}))


def _attach(message: CanonicalMessage, metadata: Mapping[str, Any], item: Any, *, preserve_raw: bool) -> None:
	"""Attach optional metadata/usage/raw payload onto a canonical message."""

	if metadata:
		setattr(message, "metadata", dict(metadata))
	usage = _usage(item)
	if usage:
		setattr(message, "token_usage", usage)
	if preserve_raw and isinstance(item, Mapping):
		setattr(message, "raw", json_ready(dict(item)))


def _detect_human_name(items: Iterable[Any]) -> str | None:
	"""Identify the human/initiator name from a legacy AG2 transcript.

	The first plain-text turn is the initiator's opening message, so its ``name``
	is the user side. Returns ``None`` for the AgentChat shape (which carries a
	reliable ``source`` and needs no perspective correction).
	"""

	for item in items:
		if get_value(item, "source", default=None) is not None:
			return None  # AgentChat shape — source is authoritative
		if not isinstance(item, Mapping):
			continue
		if item.get("tool_calls") or item.get("tool_responses") or item.get("function_call"):
			continue
		if item.get("role") in ("system", "tool", "function"):
			continue
		name = item.get("name")
		if name is not None:
			return str(name)
	return None


def messages_from_autogen_item(
	item: Any,
	*,
	human_name: str | None = None,
	preserve_raw: bool = False,
) -> list[CanonicalMessage]:
	"""Normalize one AutoGen / AG2 message or event into canonical messages."""

	item_type = _autogen_type(item)
	lower_type = item_type.lower()
	content = get_value(item, "content", default=None)
	metadata = compact_dict(
		{
			"autogen_type": item_type,
			"source": get_value(item, "source", default=None),
			"metadata": object_to_dict(get_value(item, "metadata", default={})),
		}
	)

	# 1. Tool-call requests (assistant asking to invoke a tool).
	raw_tool_calls = get_value(item, "tool_calls", "function_calls", default=None)
	is_tool_call = (
		bool(raw_tool_calls)
		or "toolcallrequest" in lower_type
		or "tool_call_request" in lower_type
		or (isinstance(item, Mapping) and item.get("role") == "tool_call")
	)
	if is_tool_call:
		calls = as_list(raw_tool_calls if raw_tool_calls else (content or []))
		messages: list[CanonicalMessage] = []
		for call in calls:
			fn = get_value(call, "function", default=None)  # OpenAI nests name/arguments here
			call_id = get_value(call, "id", "call_id", "tool_call_id", default=None) or "call_unknown"
			name = get_value(call, "name", "tool_name", default=None) or get_value(fn, "name", default=None)
			args = get_value(call, "arguments", "args", default=None)
			if args is None:
				args = get_value(fn, "arguments", default={})
			# Build with clean tool_args; carry the id/metadata on the message itself
			# so internal bookkeeping never leaks into the serialized arguments.
			message = tool_call_message(str(name) if name else None, parse_jsonish(args))
			message.tool_call_id = str(call_id)
			message.tool_calls = [function_call_to_tool_call(call)]
			if metadata:
				setattr(message, "metadata", dict(metadata))
			messages.append(message)
		return messages or [
			message_to_canonical(item, default_role="tool_call", metadata=metadata, preserve_raw=preserve_raw)
		]

	# 2. Tool-call executions / responses.
	raw_results = get_value(item, "results", "tool_results", "tool_responses", default=None)
	is_tool_response = (
		raw_results is not None
		or "toolcallexecution" in lower_type
		or "tool_call_execution" in lower_type
		or (isinstance(item, Mapping) and (item.get("role") in ("tool", "function") or "tool_call_id" in item))
	)
	if is_tool_response:
		if raw_results is not None:
			results = as_list(raw_results)
		elif content is not None:
			results = as_list(content)
		else:
			results = [item]
		messages = []
		for result in results:
			call_id = get_value(result, "tool_call_id", "call_id", "id", default="call_unknown")
			name = get_value(result, "name", "tool_name", default=None)
			result_content = get_value(result, "content", "result", "output", default=result)
			if isinstance(result_content, Mapping) and "content" in result_content:
				result_content = result_content["content"]
			result_metadata = {**metadata, "is_error": get_value(result, "is_error", default=None)}
			message = tool_response_message(
				str(name) if name else None,
				result_content,
				tool_call_id=str(call_id),
				metadata=result_metadata,
			)
			message.tool_call_id = str(call_id)
			messages.append(message)
		return messages or [
			message_to_canonical(item, default_role="tool_response", metadata=metadata, preserve_raw=preserve_raw)
		]

	# 3. System turns — `role` is absolute here, never inverted.
	role_field = get_value(item, "role", default=None)
	if role_field == "system":
		message = CanonicalMessage(role="system", content=coerce_text(content))
		_attach(message, metadata, item, preserve_raw=preserve_raw)
		return [message]

	# 4. Plain user/assistant text.
	source = get_value(item, "source", "sender", "speaker", default=None)
	name = item.get("name") if isinstance(item, Mapping) else None
	if source is not None:
		role = normalize_role(source, default="assistant")
	elif human_name is not None and name is not None:
		role = "user" if str(name) == str(human_name) else "assistant"
	elif role_field is not None:
		role = normalize_role(role_field, default="assistant")
	else:
		role = "assistant"

	message = CanonicalMessage(role=role, content=coerce_text(content))
	message_metadata = dict(metadata)
	if "streamingchunk" in lower_type or "stream_chunk" in lower_type:
		message_metadata["stream_chunk"] = True
	_attach(message, message_metadata, item, preserve_raw=preserve_raw)
	return [message]


def from_task_result(
	result: Any,
	*,
	agent: Any = None,
	metadata: Mapping[str, Any] | None = None,
	preserve_raw: bool = False,
) -> CanonicalInteraction:
	"""Normalize an AG2 ``ChatResult`` / ``RunResponse`` or compatible mapping."""

	raw_messages = as_list(get_value(result, "messages", "chat_history", default=[]))
	human_name = _detect_human_name(raw_messages)
	interaction = CanonicalInteraction(
		source_framework="autogen",
		metadata={
			"result_type": result.__class__.__name__,
			"stop_reason": get_value(result, "stop_reason", default=None),
			**dict(metadata or {}),
		},
	)
	interaction.agent = compact_dict(
		{
			"name": get_value(agent, "name", default=None),
			"description": get_value(agent, "description", default=None),
			"system_message": get_value(agent, "system_message", default=None),
		}
	)

	for item in raw_messages:
		for message in messages_from_autogen_item(item, human_name=human_name, preserve_raw=preserve_raw):
			append_unique_message(interaction, message)

	token_usage = object_to_dict(get_value(result, "models_usage", "usage", "token_usage", default={}))
	if token_usage:
		interaction.token_usage = token_usage
	if preserve_raw:
		interaction.extra["raw_result"] = object_to_dict(result) or str(result)
	return interaction


def from_chat_history(
	chat_history: Any,
	*,
	metadata: Mapping[str, Any] | None = None,
	preserve_raw: bool = False,
) -> CanonicalInteraction:
	"""Normalize a legacy AG2 ``chat_history`` transcript (a list of messages)."""

	messages = get_value(chat_history, "chat_history", "messages", default=chat_history)
	raw_list = as_list(messages)
	human_name = _detect_human_name(raw_list)

	interaction = CanonicalInteraction(
		source_framework="autogen",
		metadata={"source_shape": "chat_history", **dict(metadata or {})},
	)
	for item in raw_list:
		for message in messages_from_autogen_item(item, human_name=human_name, preserve_raw=preserve_raw):
			append_unique_message(interaction, message)

	if preserve_raw:
		interaction.extra["raw_chat_history"] = object_to_dict(chat_history) or chat_history
	return interaction


def from_stream_events(
	events: Iterable[Any],
	*,
	metadata: Mapping[str, Any] | None = None,
	preserve_raw: bool = False,
) -> CanonicalInteraction:
	"""Normalize a streamed run.

	Handles both the AgentChat shape (an event exposing ``.messages``) and the
	AG2 ``run()`` shape (a terminal ``RunCompletionEvent`` exposing
	``content.history``). Any event carrying a full transcript is expanded into
	clean turns; every event is also recorded as a span for observability.
	"""

	interaction = CanonicalInteraction(
		source_framework="autogen", metadata={"source_shape": "run_stream", **dict(metadata or {})}
	)
	for event in events:
		history = get_value(event, "messages", default=None)
		if history is None:
			history = get_nested(event, "content", "history", default=None)
		if history is not None:
			final_result = from_task_result({"messages": history}, metadata=metadata, preserve_raw=preserve_raw)
			for message in final_result.messages:
				append_unique_message(interaction, message)
			interaction.spans.extend(final_result.spans)
			continue
		interaction.spans.append(
			{"kind": "autogen.stream", "event_type": _autogen_type(event), "event": object_to_dict(event) or str(event)}
		)
		# Only lift a span event into a message when it carries plain text content
		# (e.g. an AgentChat streaming chunk). Structured AG2 run() event objects
		# are recorded as spans only, so the transcript stays free of event noise —
		# their real content arrives in the terminal RunCompletionEvent history.
		content = get_value(event, "content", default=None)
		if isinstance(content, str) and content.strip():
			for message in messages_from_autogen_item(event, preserve_raw=preserve_raw):
				append_unique_message(interaction, message)
	return interaction


__all__ = [
	"from_chat_history",
	"from_stream_events",
	"from_task_result",
	"messages_from_autogen_item",
]
