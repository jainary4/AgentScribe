"""CrewAI adapter for AgentScribe – inherits from BaseAdapter."""

from __future__ import annotations

import logging
from typing import Any, Optional

from agentscribe.adapters.base import BaseAdapter
from agentscribe.core.canonical import CanonicalInteraction, CanonicalMessage

_logger = logging.getLogger("agentscribe.crewai")


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
    """"

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
            session_id = self._resolve_session_id(context)
            interaction = self._get_or_create_interaction(context, session_id)
            self._append_messages_from_context(interaction, context)

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
            session_id = self._resolve_session_id(context)
            interaction = self._get_or_create_interaction(context, session_id)

            interaction.messages.append(
                CanonicalMessage(
                    role="tool_call",
                    content="",
                    tool_name=getattr(context, "tool_name", "unknown"),
                    tool_args=getattr(context, "tool_input", {}),
                )
            )
            result = getattr(context, "tool_result", "")
            interaction.messages.append(
                CanonicalMessage(
                    role="tool_response",
                    content=result if isinstance(result, str) else str(result),
                    tool_name=getattr(context, "tool_name", "unknown"),
                    tool_result=result if isinstance(result, str) else str(result),
                )
            )
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

        crew = getattr(context, "crew", None)
        crew_name = getattr(crew, "name", None) if crew else None
        agent = getattr(context, "agent", None)
        agent_role = getattr(agent, "role", "unknown") if agent else "unknown"
        task = getattr(context, "task", None)
        task_desc = getattr(task, "description", "unknown") if task else "unknown"
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
            crew = getattr(context, "crew", None)
            crew_name = getattr(crew, "name", None) if crew else None
            agent = getattr(context, "agent", None)
            agent_role = getattr(agent, "role", None) if agent else None
            task = getattr(context, "task", None)
            task_desc = getattr(task, "description", None) if task else None

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

        iterations = getattr(context, "iterations", 0)
        max_iter = getattr(getattr(context, "agent", None), "max_iter", None)
        if max_iter is not None and iterations >= max_iter:
            return True
        response = getattr(context, "response", "") or ""
        if "Final Answer:" in response:
            return True
        return False