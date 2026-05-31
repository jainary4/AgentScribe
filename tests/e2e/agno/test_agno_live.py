"""Live e2e test for the Agno adapter.

Runs a real Agno agent against OpenRouter and verifies the full capture
pipeline: agent run -> from_run_output -> Formatter -> write_jsonl -> read back.

Requires `agno` installed and OPENROUTER_API_KEY set; otherwise skipped.
Run with: pytest -m live tests/e2e/agno/test_agno_live.py
"""

import json
import os

import pytest

# Skip the whole module at collection time unless agno is installed, so the
# default `pytest tests/` stays green without the live dependency.
pytest.importorskip("agno")

# Marks this as a live, network/API-key test (run with `-m live`).
pytestmark = pytest.mark.live

from agno.agent import Agent
from agno.models.openrouter import OpenRouter
from agno.tools.duckduckgo import DuckDuckGoTools

from agentscribe.adapters.agno import from_run_output
from agentscribe.core.formatter import Formatter
from agentscribe.storage import write_jsonl


# Helper for key
def _require_key():
    if not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set")


@pytest.fixture(scope="module")
def agent():
    _require_key()
    # No tools, simple test
    return Agent(model=OpenRouter(id="google/gemini-3.1-flash-lite"))


@pytest.fixture(scope="module")
def search_agent():
    _require_key()
    return Agent(model=OpenRouter(id="google/gemini-3.1-flash-lite"), tools=[DuckDuckGoTools()])


def test_run_output_captures_exchange_and_writes_openai_chat_jsonl(agent, tmp_path):
    run = agent.run("Tell me a joke about Python.")
    interaction = from_run_output(run)

    # Adapter captured a real user->assistant exchange with non-empty content.
    roles = [m.role for m in interaction.messages]
    assert "user" in roles and "assistant" in roles
    assistant = [m for m in interaction.messages if m.role == "assistant"]
    assert assistant[-1].content.strip(), "assistant content is empty (run likely failed)"

    # Full pipeline: format -> write -> read back from disk.
    record = Formatter(format="openai_chat").format_single(interaction)
    output = tmp_path / "agno_training.jsonl"
    write_jsonl(str(output), [record])

    lines = output.read_text().splitlines()
    assert len(lines) == 1
    written = json.loads(lines[0])
    assert written["messages"][0]["role"] == "user"
    assert any(m["role"] == "assistant" and (m.get("content") or "").strip() for m in written["messages"])


def test_weather_run_captures_tool_call(search_agent, tmp_path):
    run = search_agent.run("What is the current weather in Toronto?")
    interaction = from_run_output(run)

    # Agno swallows DuckDuckGo failures (rate-limit/blocked) rather than raising:
    # the tool still "runs" but every response is "No results found." and the
    # assistant answer is empty. Skip — not fail — when no search actually succeeded.
    tool_responses = [m for m in interaction.messages if m.role == "tool_response"]
    succeeded = [m for m in tool_responses if (m.content or "").strip() and "No results found" not in m.content]
    if not succeeded:
        pytest.skip("DuckDuckGo returned no results (rate-limited/blocked)")

    # A real search ran: verify the tool call survives the full pipeline to JSONL.
    record = Formatter(format="openai_chat").format_single(interaction)
    output = tmp_path / "agno_weather.jsonl"
    write_jsonl(str(output), [record])

    written = json.loads(output.read_text().splitlines()[0])
    assert any(m["role"] in ("tool_call", "tool_response") for m in written["messages"])
