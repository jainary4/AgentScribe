# agentscribe/adapters/base.py
"""Base adapter class for all framework adapters.

Provides buffering, formatting, storage writing, and session tracking.
Framework‑specific adapters inherit from this class and implement
their own capture hooks / listeners.
"""

from __future__ import annotations

import threading
from typing import Any

from agentscribe.core.canonical import CanonicalInteraction
from agentscribe.core.formatter import Formatter
from agentscribe.storage import write_jsonl


class BaseAdapter:
    """Common logic for capturing agent interactions and saving them as datasets.

    Parameters
    ----------
    format : str
        Output format (``"openai_chat"``, ``"sharegpt"``, ``"alpaca"``, etc.).
    output : str
        File path or cloud URI (``s3://``, ``gs://``, ``az://``).
    flush_interval : int
        Number of completed interactions to buffer before writing to storage.
        Use 0 to flush after every interaction.
    """

    def __init__(
        self,
        format: str = "openai_chat",
        output: str = "./agentscribe_data.jsonl",
        flush_interval: int = 10,
    ) -> None:
        self._format = format
        self._output = output
        self._flush_interval = flush_interval
        self._buffer: list[CanonicalInteraction] = []
        self._lock = threading.Lock()
        self._pending: dict[str, CanonicalInteraction] = {}
        self._formatter = Formatter(format=self._format)


    def _finalise_and_flush(self, session_id: str) -> None:

        """Move a finished interaction from pending to the buffer.

        If the buffer reaches ``_flush_interval``, write everything to storage.

        Parameters
        ----------
        session_id : str
            The session whose interaction just completed.

        Notes
        -----
        Subclasses call this when they detect that an agent has finished its
        task (e.g., ``"Final Answer:"`` in CrewAI, or an ``on_chain_end``
        event in LangChain).
        """

        interaction = self._pending.pop(session_id, None)
        if interaction is None:
            return
        with self._lock:
            self._buffer.append(interaction)
            if len(self._buffer) >= self._flush_interval:
                self._flush_buffer()


    def _flush_buffer(self) -> int:

        """ Format and write all buffered interactions to storage.

        Returns
        -------
        int
            Number of interactions successfully written.

        Notes
        -----
        If the write fails, the interactions are **put back** into the buffer
        so they are not lost"""

        if not self._buffer:
            return 0

        interactions = self._buffer[:]
        self._buffer.clear()

        try:
            formatted = [self._formatter.format_single(i) for i in interactions]
            write_jsonl(self._output, formatted, mode="a")
            return len(formatted)

        except Exception:
            self._buffer = interactions + self._buffer
            return 0


    def flush(self) -> int:

         """Write all buffered interactions to storage immediately.

        Returns
        -------
        int
            Number of interactions written.

        Examples
        --------
         adapter = SomeAdapter(output="./data.jsonl")
         # … agent runs …
         adapter.flush()  # force write everything buffered so far
        5
        """
        with self._lock:
            return self._flush_buffer()

    def __enter__(self):

          """Enter a ``with`` block — no special setup needed."""

        return self

    def __exit__(self, *args):

        """Exit the ``with`` block — automatically flushes all data.

        Example
        -------
         with SomeAdapter(output="./data.jsonl") as adapter:
            crew.kickoff()
            # flush() is called automatically here
        """"
        
        self.flush()