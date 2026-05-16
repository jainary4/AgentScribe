"""Agno adapter for AgentScribe.

Captures agent interactions using Agno's native post‑hooks and tool hooks.
No additional dependencies beyond ``agno`` are required.

Usage (post‑hooks — recommended):
    from agentscribe.adapters.agno import AgnoAdapter

    adapter = AgnoAdapter(format="openai_chat", output="./data.jsonl")

    agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        tools=[YFinanceTools(stock_price=True)],
        post_hooks=[adapter.post_hook],   # captures full message history
        tool_hooks=[adapter.tool_hook],   # captures individual tool calls
    )
    agent.print_response("What is the stock price of Apple?")
    adapter.flush()  # optional — also auto‑flushed on garbage collection

Alternative: MLflow autolog (requires ``mlflow>=3.3``)
    import mlflow
    mlflow.agno.autolog()
    # AgentScribe's CLI can later convert MLflow traces to training data:
    #   agentscribe convert agno-mlflow ./mlruns/ --format openai_chat --output ./data.jsonl
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from agentscribe.adapters.base import BaseAdapter
from agentscribe.core.canonical import CanonicalInteraction, CanonicalMessage
from agentscribe.core.formatter import Formatter

_logger = logging.getLogger("agentscribe.agno")


class AgnoAdapter(BaseAdapter):
    """Capture Agno agent interactions using post‑hooks.

    Inherits buffering, formatting, and storage from :class:`BaseAdapter`.

    Parameters
    ----------
    format : str
        Output format (``"openai_chat"``, ``"sharegpt"``, ``"alpaca"``, etc.).
    output : str
        File path or cloud URI (``s3://``, ``gs://``, ``az://``).
    flush_interval : int
        Number of completed interactions to buffer before writing to storage.
        Use 0 to flush after every interaction.

    Examples
    --------
    >>> from agentscribe.adapters.agno import AgnoAdapter
    >>> from agno.agent import Agent
    >>> from agno.models.openai import OpenAIChat
    >>>
    >>> adapter = AgnoAdapter(format="openai_chat", output="./agno_data.jsonl")
    >>>
    >>> agent = Agent(
    ...     model=OpenAIChat(id="gpt-4o"),
    ...     tools=[...],
    ...     post_hooks=[adapter.post_hook],
    ...     tool_hooks=[adapter.tool_hook],
    ... )
    >>> agent.print_response("Hello world")
    >>> adapter.flush()

    Notes
    -----
    - ``post_hook`` captures the complete ``RunOutput.messages`` list after each
      agent run, including system, user, assistant, and tool messages.
    - ``tool_hook`` captures individual tool calls with arguments and results.
    - Both hooks can be used together; the adapter deduplicates tool messages.
    - For production, mark the hook with ``@hook(run_in_background=True)`` to
      avoid blocking the agent's response path (requires AgentOS).
    """

    def __init__(
        self,
        format: str = "openai_chat",
        output: str = "./agentscribe_data.jsonl",
        flush_interval: int = 10,
    ) -> None:
        # 1. Initialise the shared part (buffer, lock, formatter, etc.)
        super().__init__(
            format=format,
            output=output,
            flush_interval=flush_interval,
        )

        # 2. Track tool calls seen in tool_hook so we don't duplicate them
        #    when post_hook also sees them in RunOutput.messages
        self._seen_tool_calls: set[str] = set()

        _logger.info("AgentScribe Agno adapter initialised.")

    # ==================================================================
    # POST‑HOOK  (primary capture – runs after every agent response)
    # ==================================================================

    def post_hook(
        self,
        run_output: Any,
        agent: Any,
        session: Any = None,
        run_context: Any = None,
    ) -> None:
        """Post‑hook for Agno agents.  Pass this to ``Agent(post_hooks=[...])``.

        Called by Agno after the agent generates a response.  Receives the
        full ``RunOutput`` object, which contains the complete message history.

        Parameters
        ----------
        run_output : RunOutput
            The output from the agent run.  Contains ``messages`` (list of
            ``Message`` objects), ``content``, ``agent_name``, ``session_id``,
            ``tools``, and more.
        agent : Agent
            Reference to the Agent instance.
        session : Session, optional
            The current agent session.
        run_context : RunContext, optional
            The current run context.

        Notes
        -----
        This hook is called **once per agent.run() invocation** (not per LLM
        call).  It captures the complete conversation in one interaction.

        Example
        -------
        >>> agent = Agent(
        ...     post_hooks=[adapter.post_hook],
        ... )
        >>> agent.print_response("What is AI?")
        # post_hook fires automatically; data is buffered for writing
        """
        try:
            # ---- 1. Build a unique session identifier ----
            session_id = self._resolve_session_id(run_output, agent)

            # ---- 2. Create a CanonicalInteraction ----
            interaction = CanonicalInteraction(
                source_framework="agno",
                session_id=session_id,
                metadata={
                    "agent_name": getattr(run_output, "agent_name", None)
                                  or getattr(agent, "name", None),
                    "agent_id": getattr(run_output, "agent_id", None),
                    "run_id": getattr(run_output, "run_id", None),
                    "model": getattr(run_output, "model", None),
                    "user_id": getattr(run_output, "user_id", None),
                },
            )

            # ---- 3. Extract messages from RunOutput.messages ----
            messages = getattr(run_output, "messages", None) or []
            for msg in messages:
                self._append_message(interaction, msg)

            # ---- 4. If no messages were captured (edge case), extract from content ----
            if not interaction.messages:
                content = getattr(run_output, "content", None)
                if content:
                    interaction.messages.append(
                        CanonicalMessage(
                            role="assistant",
                            content=str(content),
                        )
                    )

            # ---- 5. Finalise immediately (Agno post‑hook = one complete run) ----
            self._finalise_one(interaction)

        except Exception as exc:
            _logger.error("Error in AgentScribe Agno post‑hook: %s", exc)

    # ==================================================================
    # TOOL HOOK  (secondary capture – runs on every tool call)
    # ==================================================================

    def tool_hook(
        self,
        function_name: str,
        function_call: Any,
        arguments: dict,
    ) -> Any:
        """Tool hook for Agno agents.  Pass this to ``Agent(tool_hooks=[...])``.

        Called by Agno before a tool is executed.  Wraps the tool call to
        capture its arguments and result.

        Parameters
        ----------
        function_name : str
            The name of the tool being called (e.g. ``"get_stock_price"``).
        function_call : Callable
            The actual tool function.
        arguments : dict
            The arguments being passed to the tool.

        Returns
        -------
        Any
            The result of calling ``function_call(**arguments)``.

        Notes
        -----
        Because this hook wraps the tool call, it captures both the arguments
        (before execution) and the result (after execution).  These are stored
        in an internal buffer and merged into the next ``post_hook`` interaction
        if they haven't already been captured via ``RunOutput.messages``.

        Example
        -------
        >>> agent = Agent(
        ...     tool_hooks=[adapter.tool_hook],
        ... )
        # Every tool call is now logged with arguments and results
        """
        import time

        start_time = time.time()

        try:
            # Execute the actual tool
            result = function_call(**arguments)
            duration = time.time() - start_time

            # Store for later merging into the CanonicalInteraction
            self._pending_tool_result = {
                "tool_name": function_name,
                "tool_args": arguments,
                "tool_result": result,
                "duration_ms": int(duration * 1000),
            }

            return result

        except Exception as exc:
            duration = time.time() - start_time
            _logger.error(
                "Tool '%s' failed after %.2fs: %s",
                function_name, duration, exc,
            )
            raise  # re‑raise so Agno can handle the error

    # ==================================================================
    # INTERNAL HELPERS
    # ==================================================================

    def _resolve_session_id(self, run_output: Any, agent: Any) -> str:
        """Build a stable session identifier from the run output and agent.

        Parameters
        ----------
        run_output : RunOutput
        agent : Agent

        Returns
        -------
        str
            e.g. ``"StockAgent:session_abc123:run_def456"``
        """
        agent_name = (
            getattr(run_output, "agent_name", None)
            or getattr(agent, "name", None)
            or "unknown"
        )
        session_id = getattr(run_output, "session_id", None)
        run_id = getattr(run_output, "run_id", None)

        parts = [agent_name]
        if session_id:
            parts.append(session_id)
        if run_id:
            parts.append(run_id)
        return ":".join(parts)

    def _append_message(
        self,
        interaction: CanonicalInteraction,
        msg: Any,
    ) -> None:
        """Convert an Agno ``Message`` object into a ``CanonicalMessage`` and
        append it to the interaction.

        Parameters
        ----------
        interaction : CanonicalInteraction
            The interaction being built.
        msg : Message
            An Agno Message object with ``role`` and ``content`` attributes.
        """
        # Extract role and content from the Agno Message object
        role = getattr(msg, "role", None)
        if role is None:
            # Fallback: try dict‑like access
            role = msg.get("role", "unknown") if isinstance(msg, dict) else "unknown"

        content = getattr(msg, "content", None)
        if content is None and isinstance(msg, dict):
            content = msg.get("content", "")

        role = str(role).lower()
        content = str(content) if content else ""

        # Map Agno‑specific role names to canonical roles
        if role in ("tool", "tool_call", "function_call"):
            # This is a tool‑call message (agent decided to call a tool)
            tool_name = getattr(msg, "tool_name", None) or "unknown"
            tool_args = getattr(msg, "tool_args", None) or {}
            interaction.messages.append(
                CanonicalMessage(
                    role="tool_call",
                    content="",
                    tool_name=tool_name,
                    tool_args=tool_args,
                )
            )
        elif role in ("tool_response", "tool_result", "function_result"):
            # This is a tool‑response message (result from tool execution)
            tool_name = getattr(msg, "tool_name", None) or "unknown"
            tool_result = str(content) if content else ""
            interaction.messages.append(
                CanonicalMessage(
                    role="tool_response",
                    content=tool_result,
                    tool_name=tool_name,
                    tool_result=tool_result,
                )
            )
        else:
            # Standard roles: system, user, assistant
            interaction.messages.append(
                CanonicalMessage(role=role, content=content)
            )

    def _finalise_one(self, interaction: CanonicalInteraction) -> None:
        """Finalise a single interaction immediately (used by post‑hook).

        Unlike the CrewAI adapter which waits for a final iteration signal,
        Agno post‑hooks fire once per complete agent run, so we can finalise
        immediately.

        Parameters
        ----------
        interaction : CanonicalInteraction
            The completed interaction to buffer and potentially flush.
        """
        with self._lock:
            self._buffer.append(interaction)
            if len(self._buffer) >= self._flush_interval:
                self._flush_buffer()


# ======================================================================
#  POST‑HOC PARSER  –  for CLI log conversion
# ======================================================================

def parse_agno_run_output(
    run_output: Any,
    format_name: str = "openai_chat",
) -> list[dict[str, Any]]:
    """Parse an Agno ``RunOutput`` object (or its JSON serialisation) into
    formatted training records.

    Useful for post‑hoc batch conversion when you have saved ``RunOutput``
    objects from previous agent runs.

    Parameters
    ----------
    run_output : RunOutput or dict
        An Agno RunOutput object, or a dictionary with the same structure.
    format_name : str
        Desired output format.

    Returns
    -------
    list[dict]
        Formatted records ready for writing to JSONL.

    Example
    -------
    >>> from agno.agent import Agent
    >>> agent = Agent(model=...)
    >>> run_output = agent.run("Hello")
    >>> records = parse_agno_run_output(run_output, format_name="sharegpt")
    >>> # records is a list of dicts ready for write_jsonl
    """

    # If run_output is a dict, work with it directly; otherwise use getattr
    if isinstance(run_output, dict):
        messages = run_output.get("messages", [])
        agent_name = run_output.get("agent_name", "")
        session_id = run_output.get("session_id", "")
        run_id = run_output.get("run_id", "")
        model = run_output.get("model", "")
        agent_id = run_output.get("agent_id", "")
        user_id = run_output.get("user_id", "")
    else:
        messages = getattr(run_output, "messages", None) or []
        agent_name = getattr(run_output, "agent_name", None) or ""
        session_id = getattr(run_output, "session_id", None) or ""
        run_id = getattr(run_output, "run_id", None) or ""
        model = getattr(run_output, "model", None) or ""
        agent_id = getattr(run_output, "agent_id", None) or ""
        user_id = getattr(run_output, "user_id", None) or ""

    # Build the canonical interaction
    interaction = CanonicalInteraction(
        source_framework="agno",
        session_id=f"{agent_name}:{session_id}:{run_id}",
        metadata={
            "agent_name": agent_name,
            "agent_id": agent_id,
            "run_id": run_id,
            "model": model,
            "user_id": user_id,
        },
    )

    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            tool_name = msg.get("tool_name")
            tool_args = msg.get("tool_args")
            tool_result = msg.get("tool_result")
        else:
            role = getattr(msg, "role", "unknown")
            content = getattr(msg, "content", "")
            tool_name = getattr(msg, "tool_name", None)
            tool_args = getattr(msg, "tool_args", None)
            tool_result = getattr(msg, "tool_result", None)

        role = str(role).lower()

        if role in ("tool", "tool_call", "function_call"):
            interaction.messages.append(
                CanonicalMessage(
                    role="tool_call",
                    content="",
                    tool_name=tool_name or "unknown",
                    tool_args=tool_args or {},
                )
            )
        elif role in ("tool_response", "tool_result", "function_result"):
            interaction.messages.append(
                CanonicalMessage(
                    role="tool_response",
                    content=str(content) if content else "",
                    tool_name=tool_name or "unknown",
                    tool_result=str(tool_result) if tool_result else "",
                )
            )
        else:
            interaction.messages.append(
                CanonicalMessage(role=role, content=str(content) if content else "")
            )

    formatter = Formatter(format=format_name)
    return [formatter.format_single(interaction)]