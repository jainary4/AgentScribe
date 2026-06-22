
"""
Tests for the AgentScribe Pydantic AI adapter.
 
Fully offline and deterministic: uses Pydantic AI's TestModel, so there is no API
key, no network, and no nondeterministic LLM output. TestModel calls each registered
tool once and then returns, which is exactly the transcript shape we want to capture.
 
Run:  python test_pydantic_ai.py
"""
 
from __future__ import annotations
 
import json
import os
import tempfile
 
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
import pytest
 
from agentscribe.adapters.pydantic_ai import (
    from_run,
    from_session,
    PydanticAITraceCollector,
    PydanticAIAdapter,
)
from agentscribe.core.formatter import Formatter
from agentscribe.storage import write_jsonl

OUT = Path("./out"); OUT.mkdir(exist_ok=True)
 
# --------------------------------------------------------------------------- #
# Agents (offline)
# --------------------------------------------------------------------------- #
def make_chat_agent() -> Agent:
    return Agent(TestModel(), instructions="You are a concise assistant.")
 
 
def make_tool_agent() -> Agent:
    agent = Agent(TestModel(), instructions="You answer questions using tools.")
 
    @agent.tool_plain
    def calculator(expression: str) -> str:
        """Evaluate an arithmetic expression."""
        return "42"
 
    @agent.tool_plain
    def search(query: str) -> str:
        """Search for information."""
        return "(search) a mock result"
 
    return agent
 
 
# --------------------------------------------------------------------------- #
# OpenAI fine-tuning structural validator (same checks as the Atomic suite)
# --------------------------------------------------------------------------- #
def validate_openai_line(obj: dict) -> list[str]:
    errs: list[str] = []
    msgs = obj.get("messages") or []
    open_calls: dict[str, str] = {}
    for m in msgs:
        role = m.get("role")
        if role == "assistant" and "tool_calls" in m:
            if not isinstance(m.get("content"), str):
                errs.append("assistant.content must be a string, not null")
            for c in m["tool_calls"]:
                open_calls[c["id"]] = c["function"]["name"]
                try:
                    json.loads(c["function"]["arguments"])
                except Exception:  # noqa: BLE001
                    errs.append(f"tool_call {c['id']} arguments not valid JSON")
        elif role == "tool":
            if m.get("tool_call_id") not in open_calls:
                errs.append(f"tool result {m.get('tool_call_id')} has no matching call")
            if not isinstance(m.get("content"), str):
                errs.append("tool.content must be a string")
        elif role in ("system", "user", "assistant"):
            if not isinstance(m.get("content"), str):
                errs.append(f"{role}.content must be a string")
    return errs
    
 
 
def roles_of(interaction) -> list[str]:
    return [m.role for m in interaction.messages]
 
 
# --------------------------------------------------------------------------- #
# TEST 1 — from_run on a plain chat agent (no tools)
# --------------------------------------------------------------------------- #
def test_from_run_chat() -> None:
    print("\n[TEST 1] from_run: chat agent (no tools)")
    agent = make_chat_agent()
    interaction = from_run(agent.run_sync("Tell me a fact about the moon."))
    oai = Formatter("openai_chat").format_single(interaction)
    roles = [m["role"] for m in oai["messages"]]
    print("   roles:", roles, "| model:", interaction.metadata.get("model"))
    assert "user" in roles and "assistant" in roles
    assert validate_openai_line(oai) == []
    print("OK")
    write_jsonl(OUT/"m1.jsonl",[oai] mode=a)
 
 
# --------------------------------------------------------------------------- #
# TEST 2 — from_run on a tool agent -> real tool_calls
# --------------------------------------------------------------------------- #
def test_from_run_tools() -> None:
    print("\n[TEST 2] from_run: tool agent -> structured tool_calls")
    agent = make_tool_agent()
    interaction = from_run(agent.run_sync("What is 21 * 2, and find a fact about it?"))
    roles = roles_of(interaction)
    print("   canonical roles:", roles)
    assert "tool_call" in roles and "tool_response" in roles
 
    oai = Formatter("openai_chat").format_single(interaction)
    write_jsonl(OUT/"m2.jsonl",[oai] mode=a)

    errs = validate_openai_line(oai)
    n_calls = sum(len(m.get("tool_calls", [])) for m in oai["messages"])
    print("   openai_chat tool_calls:", n_calls, "| valid:", errs or "YES")
    assert errs == []
    assert n_calls >= 2                      # TestModel calls both tools
    # every tool result id matches a call id
    call_ids = {c["id"] for m in oai["messages"] for c in m.get("tool_calls", [])}
    tool_ids = {m["tool_call_id"] for m in oai["messages"] if m["role"] == "tool"}
    assert tool_ids <= call_ids
    print("   OK: tool ids all matched, no nulls")
 
 
# --------------------------------------------------------------------------- #
# TEST 3 — from_session captures the whole multi-turn conversation
# --------------------------------------------------------------------------- #
def test_from_session_multiturn() -> None:
    print("\n[TEST 3] from_session: multi-turn conversation")
    agent = make_chat_agent()
    r1 = agent.run_sync("Hi!")
    r2 = agent.run_sync("And again?", message_history=r1.all_messages())
 
    one_run = from_run(r2)                    # just the second run
    whole = from_session(r2)                  # both runs (all_messages)
    print("   from_run turns:", len(one_run.messages), "| from_session turns:", len(whole.messages))
    assert len(whole.messages) > len(one_run.messages)
    assert validate_openai_line(Formatter("openai_chat").format_single(whole)) == []
    print("   OK: session spans more turns than a single run")
 
 
# --------------------------------------------------------------------------- #
# TEST 4 — collector batches many runs, one write
# --------------------------------------------------------------------------- #
def test_collector_batch(tmp: str) -> None:
    print("\n[TEST 4] PydanticAITraceCollector: batch")
    agent = make_tool_agent()
    out = os.path.join(tmp, "collector.jsonl")
    c = PydanticAITraceCollector(format_name="openai_chat", output_path=out)
    for q in ["q1", "q2", "q3"]:
        c.record_run(agent.run_sync(q))
    n = c.flush(append=False)
    lines = [json.loads(l) for l in open(out)]
    print("   recorded 3, flush wrote", n, "lines:", len(lines))
    assert n == 3 and len(lines) == 3
    assert all(validate_openai_line(l) == [] for l in lines)
    print("   OK")
 
 
# --------------------------------------------------------------------------- #
# TEST 5 — adapter captures live, flushes
# --------------------------------------------------------------------------- #
def test_adapter_live(tmp: str) -> None:
    print("\n[TEST 5] PydanticAIAdapter: live capture")
    agent = make_chat_agent()
    out = os.path.join(tmp, "adapter.jsonl")
    adapter = PydanticAIAdapter(format="openai_chat", output=out, flush_interval=0)  # write each
    adapter.capture(agent.run_sync("one"))
    adapter.capture(agent.run_sync("two"))
    lines = sum(1 for _ in open(out))
    print("   lines after 2 captures (flush_interval=0):", lines)
    assert lines == 2
    print("   OK")
 
 
# --------------------------------------------------------------------------- #
# TEST 6 — one capture -> many formats
# --------------------------------------------------------------------------- #
def test_multi_format() -> None:
    print("\n[TEST 6] one capture -> openai_chat / sharegpt / alpaca / prompt_completion")
    agent = make_tool_agent()
    interaction = from_run(agent.run_sync("compute and search"))
    for fmt in ["openai_chat", "sharegpt", "alpaca", "prompt_completion"]:
        record = Formatter(fmt).format_single(interaction)
          write_jsonl(OUT/"m6.jsonl",[oai] mode=a)
        assert isinstance(record, dict) and record
        print(f"   {fmt:18} -> keys {list(record.keys())}")
    print("   OK")
 
 
if __name__ == "__main__":
    test_from_run_chat()
    test_from_run_tools()
    test_from_session_multiturn()
    with tempfile.TemporaryDirectory() as tmp:
        test_collector_batch(tmp)
        test_adapter_live(tmp)
    test_multi_format()
    print("\nAll tests passed.")
 