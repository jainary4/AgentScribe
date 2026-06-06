"""CrewAI adapter for AgentScribe – inherits from BaseAdapter."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Optional

from agentscribe.adapters.base import BaseAdapter
from agentscribe.adapters.utils import (
    append_unique_message,
    build_metadata,
    compact_dict,
    get_value,
    interaction_from_messages,
    message_to_canonical,
    object_to_dict,
    resolve_identifier,
    serialize_object_list,
    tool_call_message,
    tool_response_message,
)
from agentscribe.core.canonical import CanonicalInteraction, CanonicalMessage

_logger = logging.getLogger("agentscribe.crewai")


def _agent_metadata(agent: Any) -> dict[str, Any]:
    """Build serialisable CrewAI agent metadata.

    Parameters
    ----------
    agent : Any
        CrewAI agent object or mapping.

    Returns
    -------
    dict[str, Any]
        Compact metadata describing the agent, model, and tools.
    """

    return build_metadata(
        agent,
        fields={
            "name": ("name", "role"),
            "role": ("role",),
            "goal": ("goal",),
            "backstory": ("backstory",),
            "model": ("llm", "model"),
            "tools": lambda value: serialize_object_list(get_value(value, "tools", default=[]) or []),
        },
    )


def _task_metadata(task: Any) -> dict[str, Any]:
    """Build serialisable CrewAI task metadata.

    Parameters
    ----------
    task : Any
        CrewAI task object or mapping.

    Returns
    -------
    dict[str, Any]
        Compact metadata describing the task.
    """

    return build_metadata(
        task,
        fields={
            "description": ("description",),
            "expected_output": ("expected_output",),
            "name": ("name",),
            "id": ("id",),
        },
    )


def _context_session_id(context: Any) -> str | None:
    """Resolve a session id from a CrewAI hook context.

    Parameters
    ----------
    context : Any
        CrewAI hook context or compatible mapping.

    Returns
    -------
    str | None
        Session identifier when one is present.
    """

    return resolve_identifier(context, "session_id", "crew_id")


def _context_provenance(context: Any) -> dict[str, Any]:
    """Resolve run provenance from a CrewAI hook context.

    Parameters
    ----------
    context : Any
        CrewAI hook context or compatible mapping.

    Returns
    -------
    dict[str, Any]
        Compact provenance metadata.
    """

    return build_metadata(context, fields={"run_id": ("run_id", "execution_id", "id")})


def from_llm_call_context(context: Any, *, metadata: Mapping[str, Any] | None = None) -> CanonicalInteraction:
    """Normalize a CrewAI after-LLM-call hook context.

    Parameters
    ----------
    context : Any
        CrewAI LLM call hook context or compatible mapping.
    metadata : Mapping[str, Any] | None
        Optional metadata to merge into the canonical interaction.

    Returns
    -------
    CanonicalInteraction
        Interaction containing messages, response, and CrewAI provenance.
    """

    messages = get_value(context, "messages", "chat_history", default=[]) or []
    response = get_value(context, "response", "output", "result", "raw", default=None)
    interaction = interaction_from_messages(
        messages,
        source_framework="crewai",
        session_id=_context_session_id(context),
        metadata={"event": "llm_call", **_context_provenance(context), **dict(metadata or {})},
    )
    interaction.agent = _agent_metadata(get_value(context, "agent", default=None))
    task = _task_metadata(get_value(context, "task", default=None))
    if task:
        interaction.metadata["task"] = task
    interaction.instantiation = compact_dict(
        {
            "agent": interaction.agent,
            "task": task,
            "iterations": get_value(context, "iterations", "iteration", default=None),
        }
    )
    model = get_value(context, "model", "llm", default=None)
    if model is not None:
        interaction.model = str(model)
    token_usage = object_to_dict(get_value(context, "token_usage", "usage", default={}))
    if token_usage:
        interaction.token_usage = token_usage
    if response is not None:
        append_unique_message(
            interaction,
            message_to_canonical(
                {"role": "assistant", "content": response, "type": "crewai_llm_response"},
                metadata={"event": "llm_call"},
            ),
        )
    return interaction


def from_tool_call_context(context: Any, *, metadata: Mapping[str, Any] | None = None) -> CanonicalInteraction:
    """Normalize a CrewAI after-tool-call hook context.

    Parameters
    ----------
    context : Any
        CrewAI tool call hook context or compatible mapping.
    metadata : Mapping[str, Any] | None
        Optional metadata to merge into the canonical interaction.

    Returns
    -------
    CanonicalInteraction
        Interaction containing the tool call and tool response messages.
    """

    tool_name = get_value(context, "tool_name", "name", default=None)
    tool_args = get_value(context, "tool_input", "tool_args", "arguments", "input", default=None)
    tool_result = get_value(context, "tool_result", "result", "output", default=None)
    tool_call_id = get_value(context, "tool_call_id", "call_id", "id", default=None)
    interaction = CanonicalInteraction(
        source_framework="crewai",
        session_id=_context_session_id(context),
        metadata={"event": "tool_call", **_context_provenance(context), **dict(metadata or {})},
    )
    interaction.agent = _agent_metadata(get_value(context, "agent", default=None))
    task = _task_metadata(get_value(context, "task", default=None))
    if task:
        interaction.metadata["task"] = task
    interaction.messages.append(
        tool_call_message(str(tool_name) if tool_name is not None else None, tool_args, tool_call_id=str(tool_call_id) if tool_call_id else None)
    )
    interaction.messages.append(
        tool_response_message(
            str(tool_name) if tool_name is not None else None,
            tool_result,
            tool_call_id=str(tool_call_id) if tool_call_id else None,
        )
    )
    return interaction


def from_event(source: Any, event: Any | None = None, *, metadata: Mapping[str, Any] | None = None) -> CanonicalInteraction:
    """Normalize a CrewAI event-bus event into one canonical interaction.

    Parameters
    ----------
    source : Any
        Event source, or the event itself when ``event`` is omitted.
    event : Any | None
        CrewAI event object or mapping.
    metadata : Mapping[str, Any] | None
        Optional metadata to merge into the canonical interaction.

    Returns
    -------
    CanonicalInteraction
        Interaction inferred from the event payload.
    """

    if event is None:
        event = source
        source = get_value(event, "source", default=None)

    event_type = str(get_value(event, "type", "event_type", default=event.__class__.__name__)).lower()
    event_metadata = {
        "event_type": event_type,
        "source": object_to_dict(source) or str(source) if source is not None else None,
        **dict(metadata or {}),
    }
    if "tool" in event_type:
        return from_tool_call_context(event, metadata=event_metadata)
    if "llm" in event_type or "agent" in event_type or "task" in event_type:
        return from_llm_call_context(event, metadata=event_metadata)

    interaction = CanonicalInteraction(
        source_framework="crewai",
        session_id=_context_session_id(event),
        metadata=compact_dict(event_metadata),
    )
    interaction.spans.append({"kind": "crewai.event", "event": object_to_dict(event) or str(event)})
    user_input = get_value(event, "input", "prompt", default=None)
    output = get_value(event, "output", "response", "result", "raw", default=None)
    if user_input is not None:
        interaction.add_message("user", user_input)
    if output is not None:
        interaction.add_message("assistant", output)
    return interaction


def from_kickoff_output(output: Any, *, metadata: Mapping[str, Any] | None = None) -> CanonicalInteraction:
    """Normalize Crew.kickoff() output when only final results are available.

    Parameters
    ----------
    output : Any
        CrewAI kickoff return object or compatible mapping.
    metadata : Mapping[str, Any] | None
        Optional metadata to merge into the canonical interaction.

    Returns
    -------
    CanonicalInteraction
        Interaction containing the final assistant output.
    """

    interaction = CanonicalInteraction(source_framework="crewai", metadata={"event": "kickoff_output", **dict(metadata or {})})
    raw_output = get_value(output, "raw", "output", "result", default=output)
    interaction.add_message("assistant", raw_output)
    token_usage = object_to_dict(get_value(output, "token_usage", "usage", default={}))
    if token_usage:
        interaction.token_usage = token_usage
    tasks_output = get_value(output, "tasks_output", default=None)
    if tasks_output is not None:
        interaction.extra["tasks_output"] = [object_to_dict(task_output) or str(task_output) for task_output in tasks_output]
    return interaction


class CrewAIAdapter(BaseAdapter):
    """Capture CrewAI agent interactions using execution hooks.

    Inherits buffering, formatting, and storage from :class:`BaseAdapter`.

    Parameters
    ----------
    format : str
        Output format (``"openai_chat"``, ``"sharegpt"``, etc.).
    output : str
        File path or cloud URI.
    flush_interval : int
        Number of interactions to buffer before writing.

    Examples
    --------
    from agentscribe.adapters.crewai import CrewAIAdapter
    capture = CrewAIAdapter(format="sharegpt", output="./data.jsonl")
    crew = Crew(agents=[...], tasks=[...])
    crew.kickoff()
    capture.flush()  # optional — also flushed when the script ends
    """

    def __init__(
        self,
        format: str = "openai_chat",
        output: str = "./agentscribe_data.jsonl",
        flush_interval: int = 10,
    ) -> None:
        super().__init__(
            format=format,
            output=output,
            flush_interval=flush_interval,
        )
        self._register_hooks()

    # ------------------------------------------------------------------
    # Hook registration (CrewAI‑specific)
    # ------------------------------------------------------------------
    def _register_hooks(self) -> None:

        """Notes
        -----
        These hooks fire for **every** agent in the process — no per‑agent
        configuration needed.
        """
        try:
            from crewai.hooks import (
                register_after_llm_call_hook,
                register_after_tool_call_hook,
            )

            register_after_llm_call_hook(self._on_after_llm)
            register_after_tool_call_hook(self._on_after_tool)

            _logger.info("AgentScribe CrewAI hooks registered.")
        except ImportError:
            _logger.warning(
                "CrewAI not found. Install crewai>=1.0 or use a different adapter."
            )
        except Exception as exc:
            _logger.error("Failed to register CrewAI hooks: %s", exc)

    # ------------------------------------------------------------------
    # LLM hook handler
    # ------------------------------------------------------------------
    def _on_after_llm(self, context: Any) -> Optional[str]:

        """Called by CrewAI after every LLM response.

        Parameters
        ----------
        context : LLMCallHookContext
            Contains ``messages``, ``response``, ``agent``, ``task``,
            ``iterations``, ``crew``.

        Returns
        -------
        str | None
            Always ``None`` — we never modify the agent’s output.
        """

        try:
            captured = from_llm_call_context(context)
            session_id = captured.session_id or self._resolve_session_id(context)
            captured.session_id = session_id
            interaction = self._get_or_create_interaction(context, session_id)
            self._merge_interaction(interaction, captured)

            if self._is_final_iteration(context):
                self._finalise_and_flush(session_id)   # inherited from BaseAdapter
        except Exception as exc:
            _logger.error("Error in AgentScribe LLM hook: %s", exc)
        return None

    # ------------------------------------------------------------------
    # Tool hook handler
    # ------------------------------------------------------------------
    def _on_after_tool(self, context: Any) -> Optional[str]:

        """Called by CrewAI after every tool execution.

        Parameters
        ----------
        context : ToolCallHookContext
            Contains ``tool_name``, ``tool_input``, ``tool_result``,
            ``agent``, ``task``.

        Returns
        -------
        str | None
            Always ``None``.
        """

        try:
            captured = from_tool_call_context(context)
            session_id = captured.session_id or self._resolve_session_id(context)
            captured.session_id = session_id
            interaction = self._get_or_create_interaction(context, session_id)
            self._merge_interaction(interaction, captured)
        except Exception as exc:
            _logger.error("Error in AgentScribe tool hook: %s", exc)
        return None

    # ------------------------------------------------------------------
    # Helpers (framework‑specific but self‑contained)
    # ------------------------------------------------------------------
    def _resolve_session_id(self, context: Any) -> str:

        """Build a stable session identifier from crew, agent, and task.

        Parameters
        ----------
        context : hook context (LLM or tool)

        Returns
        -------
        str
        e.g. ``"SupportCrew:SupportAgent:Handle password reset"``
        """

        explicit_session_id = _context_session_id(context)
        if explicit_session_id is not None:
            return explicit_session_id

        crew = get_value(context, "crew", default=None)
        crew_name = get_value(crew, "name", default=None) if crew else None
        agent = get_value(context, "agent", default=None)
        agent_role = get_value(agent, "role", "name", default="unknown") if agent else "unknown"
        task = get_value(context, "task", default=None)
        task_desc = get_value(task, "description", "name", default="unknown") if task else "unknown"
        if crew_name:
            return f"{crew_name}:{agent_role}:{task_desc}"
        return f"{agent_role}:{task_desc}"

    def _get_or_create_interaction(self, context: Any, session_id: str) -> CanonicalInteraction:

        """Return the existing interaction for this session, or create a new one.

        Parameters
        ----------
        context : hook context
        session_id : str

        Returns
        -------
        CanonicalInteraction
        """

        if session_id not in self._pending:
            crew = get_value(context, "crew", default=None)
            crew_name = get_value(crew, "name", default=None) if crew else None
            agent = get_value(context, "agent", default=None)
            agent_role = get_value(agent, "role", "name", default=None) if agent else None
            task = get_value(context, "task", default=None)
            task_desc = get_value(task, "description", "name", default=None) if task else None

            self._pending[session_id] = CanonicalInteraction(
                source_framework="crewai",
                session_id=session_id,
                metadata={
                    "crew_name": crew_name,
                    "agent_role": agent_role,
                    "task_description": task_desc,
                },
            )
        return self._pending[session_id]

    def _merge_interaction(self, target: CanonicalInteraction, source: CanonicalInteraction) -> None:
        """Merge converter output into an in-progress live hook interaction."""

        for message in source.messages:
            append_unique_message(target, message)
        target.metadata.update(compact_dict(source.metadata))
        if source.agent:
            target.agent = source.agent
        if source.instantiation:
            target.instantiation.update(source.instantiation)
        if source.model and not target.model:
            target.model = source.model
        if source.token_usage:
            target.token_usage.update(source.token_usage)
        if source.spans:
            target.spans.extend(source.spans)

    def _append_messages_from_context(self, interaction: CanonicalInteraction, context: Any) -> None:

        """Extract new messages from the LLM hook context and add them.

        Only messages that are **not** already in the interaction are appended.
        Tool messages are skipped because the tool hook handles them separately.
        The new assistant response (``context.response``) is appended if it is
        not already the last message.

        Parameters
        ----------
        interaction : CanonicalInteraction
            The interaction being built.
        context : LLMCallHookContext
        """

        messages = getattr(context, "messages", [])
        existing_count = len(interaction.messages)

        for msg in messages[existing_count:]:
            role = msg.get("role", "unknown") if isinstance(msg, dict) else "unknown"
            content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
            if role in ("tool", "tool_call", "tool_response"):
                continue
            interaction.messages.append(
                CanonicalMessage(role=role, content=content)
            )

        response = getattr(context, "response", None)
        if response and (
            not interaction.messages
            or interaction.messages[-1].role != "assistant"
            or interaction.messages[-1].content != response
        ):
            interaction.messages.append(
                CanonicalMessage(role="assistant", content=response)
            )

    def _is_final_iteration(self, context: Any) -> bool:

        """Heuristic to detect whether the agent has finished its task.

        Returns True if:
        - The iteration count has reached the agent's ``max_iter`` limit.
        - The response contains the string ``"Final Answer:"``.

        Parameters
        ----------
        context : LLMCallHookContext

        Returns
        -------
        bool
        """

        iterations = get_value(context, "iterations", "iteration", default=0)
        max_iter = get_value(get_value(context, "agent", default=None), "max_iter", default=None)
        if max_iter is not None:
            try:
                if int(iterations or 0) >= int(max_iter):
                    return True
            except (TypeError, ValueError):
                pass
        response = get_value(context, "response", "output", "result", "raw", default="") or ""
        if "Final Answer:" in response:
            return True
        return False
        
    def flush(self) -> int:
        """Flush like BaseAdapter, but first finalize any in-progress interactions.

        CrewAI delivers a conversation as fragments across many hook fires, so an
        interaction sits in ``self._pending`` until the run's completion is
        detected. If that detection doesn't fire (e.g. the ``"Final Answer:"``
        heuristic fails on a CrewAI version), the interaction would otherwise be
        stuck in ``_pending`` forever and never written. Here we move every
        pending interaction into the buffer before flushing, so flush() / the
        ``with`` block / the atexit net never lose in-progress data.
        """
        with self._lock:
            for session_id in list(self._pending.keys()):
                self._buffer.append(self._pending.pop(session_id))
            return self._flush_buffer()


__all__ = [
    "CrewAIAdapter",
    "from_event",
    "from_kickoff_output",
    "from_llm_call_context",
    "from_tool_call_context",
]
