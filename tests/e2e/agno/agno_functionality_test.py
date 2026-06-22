"""
AgentScribe x Agno — capture methods + Alpaca format verification.

Some agents get `instructions=` (a system prompt), some don't, so we can verify
the Alpaca mapping in BOTH cases:
  * instructed agent -> default: system prompt -> `system` field, user -> instruction
                        flag on: system prompt -> instruction, user -> input
  * plain agent      -> instruction = user msg, input = "", no `system`

Flip ALPACA_SYSTEM_AS_INSTRUCTION to test the classic instruction/input mapping.
NOTE: methods 5/6/7 format INTERNALLY (AgnoAdapter / collector / parse_agno_run_output)
so they use the DEFAULT alpaca behavior — the flag only affects methods 1-4, which
call Formatter explicitly. That's fine for verifying both shapes.
"""

import json
from pathlib import Path

from agno.agent import Agent
from agno.models.openrouter import OpenRouter
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.db.sqlite import SqliteDb
from agno.tools.yfinance import YFinanceTools

from agentscribe.core.formatter import Formatter
from agentscribe.storage import write_jsonl
from agentscribe.adapters.agno import (
    from_run_output, from_session, from_event_stream, from_trace,
    AgnoAdapter, AgnoTraceCollector, parse_agno_run_output,
)

OUT = Path("./out"); OUT.mkdir(exist_ok=True)
MODEL = "google/gemini-3.1-flash-lite"
FORMAT = "openai_chat"
ALPACA_SYSTEM_AS_INSTRUCTION = False  # True -> classic instruction/input mapping

# System prompts for the INSTRUCTED agents:
EXTRACT_LANGS = "Use both DuckDuckGo and YFinanceTools to give me a summary on the topic the user asks"
CONCISE       = "You are a research agent. Always give 2 paragraph summaries of the research topic "

def make_agent(tools=False, instructions=None):
    return Agent(model=OpenRouter(id=MODEL),
                 tools=[DuckDuckGoTools(),YFinanceTools()] if tools else None,
                 instructions=instructions)

def fmt():
    return Formatter(FORMAT, alpaca_system_as_instruction=ALPACA_SYSTEM_AS_INSTRUCTION)


# 1. from_run_output  — INSTRUCTED agent (task in system prompt, data in user msg)
def method_1_run_output():
    agent = make_agent(instructions=EXTRACT_LANGS)
    run = agent.run("Give me anlysis on the nvidia stock and how its performance correlates to the performance of AMD stock ")
    write_jsonl(OUT / "m1_run_output.jsonl", [fmt().format_single(from_run_output(run))], mode="a")


# 2. from_session — PLAIN agent (no instructions) -> instruction=user, no system
def method_2_session():
    db = SqliteDb(db_file=str(OUT / "agno_sessions.db"))
    agent = Agent(model=OpenRouter(id=MODEL), db=db)          # no instructions
    sid = "demo-session-1"
    agent.run("Hi, my name is Aryan.", session_id=sid)
    agent.run("Give me one fun fact about octopuses.", session_id=sid)
    interactions = from_session(agent.get_session(session_id=sid))
    write_jsonl(OUT / "m2_session.jsonl", [fmt().format_single(i) for i in interactions], mode="a")


# 3. from_event_stream — PLAIN agent (excluded from docs; kept for coverage)
def method_3_event_stream():
    agent = make_agent(tools=True)
    events = agent.run("Search the weather in Toronto and summarize it.", stream=True, stream_events=True)
    write_jsonl(OUT / "m3_event_stream.jsonl", [fmt().format_single(from_event_stream(events))], mode="a")


# 4. from_trace — trace carries a system message -> behaves like an INSTRUCTED agent
EXAMPLE_OTEL_TRACE = {
    "trace_id": "demo-trace-0001",
    "spans": [{
        "name": "chat gpt-4o", "span_id": "span-llm-1", "trace_id": "demo-trace-0001",
        "attributes": {
            "openinference.span.kind": "LLM",
            "gen_ai.request.model": "gpt-4o",
            "input.value": json.dumps({"messages": [
                {"role": "system", "content": "You are a concise assistant. Answer in one short sentence."},
                {"role": "user", "content": "Give me one fact about the moon."},
            ]}),
            "output.value": json.dumps({"role": "assistant",
                "content": "The Moon drifts about 3.8 cm farther from Earth each year."}),
        },
    }],
}
def method_4_trace():
    write_jsonl(OUT / "m4_trace.jsonl", [fmt().format_single(from_trace(EXAMPLE_OTEL_TRACE))], mode="a")


# 5. AgnoAdapter hooks — INSTRUCTED agent (formats internally -> default alpaca)
def method_5_live_hooks():
    with AgnoAdapter(format=FORMAT, output=str(OUT / "m5_live.jsonl"), flush_interval=0) as capture:
        agent = Agent(model=OpenRouter(id=MODEL), tools=[DuckDuckGoTools()],
                      instructions=CONCISE,
                      post_hooks=[capture.post_hook], tool_hooks=[capture.tool_hook])
        for i in range(10):
            agent.run("What is the current weather in Toronto?")
            agent.run("Tell me a joke about Python.")


# 6. AgnoTraceCollector — MIX: one instructed run, one plain run, one trace
def method_6_collector():
    collector = AgnoTraceCollector(format_name=FORMAT, output_path=str(OUT / "m6_collector.jsonl"))
    instructed = make_agent(instructions=EXTRACT_LANGS)
    plain = make_agent()
    collector.record_run_output(instructed.run("We used Python and Go heavily, plus a bit of Ruby."))
    collector.record_run_output(plain.run("Tell me a Python joke."))
    collector.record_trace(EXAMPLE_OTEL_TRACE)
    collector.flush(append=True)


# 7. parse_agno_run_output — PLAIN agent (no instructions)
def method_7_legacy():
    agent = make_agent()
    run = agent.run("Translate 'good morning' to French.")
    write_jsonl(OUT / "m7_legacy.jsonl", parse_agno_run_output(run, format_name=FORMAT), mode="a")


# ---------------------------------------------------------------- verification
def verify_alpaca():
    print("\n=== Alpaca structure check ===")
    for p in sorted(OUT.glob("*.jsonl")):
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if not line.strip():
                continue
            r = json.loads(line)
            if "instruction" not in r:
                print(f"  [SKIP] {p.name} rec{i}: not alpaca (keys={list(r)})")
                continue
            issues = []
            if not str(r.get("instruction", "")).strip(): issues.append("empty instruction")
            if not str(r.get("output", "")).strip():      issues.append("empty output")
            tag = "OK" if not issues else "ISSUES: " + ", ".join(issues)
            print(f"  [{tag}] {p.name} rec{i}: "
                  f"system={'yes' if 'system' in r else 'no'}, "
                  f"input={'set' if r.get('input') else 'empty'}, "
                  f"history={'yes' if 'history' in r else 'no'}")
            print(f"        instruction={str(r['instruction'])[:60]!r}")
            if "system" in r:
                print(f"        system     ={str(r['system'])[:60]!r}")


if __name__ == "__main__":
    tests = [
        ("1  from_run_output (instructed)", method_1_run_output),
        ("2  from_session (plain)",         method_2_session),
        ("3  from_event_stream (plain)",    method_3_event_stream),
        ("4  from_trace (has system)",      method_4_trace),
        ("5  AgnoAdapter hooks (instructed)", method_5_live_hooks),
        ("6  AgnoTraceCollector (mixed)",   method_6_collector),
        ("7  parse_agno_run_output (plain)", method_7_legacy),
    ]
    for name, fn in tests:
        try:
            fn(); print(f"[ok]   {name}")
        except Exception as exc:
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
    verify_alpaca()