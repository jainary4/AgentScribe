"""
AgentScribe x CrewAI — every capture method against a LIVE crew.

Each method fires a real CrewAI crew and captures its response through that
method (no hardcoded data), then formats and writes it. Method 6 repeats one
method's real capture to MinIO + Postgres.

Needs: crewai, OPENROUTER_API_KEY. Each method runs its own crew (5 LLM runs).
"""

import json
from pathlib import Path

from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from crewai.hooks import (
    register_after_llm_call_hook, unregister_after_llm_call_hook,
    register_after_tool_call_hook, unregister_after_tool_call_hook,
)
from crewai.events import crewai_event_bus, LLMCallCompletedEvent

from agentscribe.core.formatter import Formatter
from agentscribe.storage import write_jsonl
from agentscribe.adapters.crewai import (
    CrewAIAdapter, from_llm_call_context, from_tool_call_context,
    from_event, from_kickoff_output,
)

OUT = Path("./out_crewai"); OUT.mkdir(exist_ok=True)
FORMAT = "openai_chat"
MODEL = "openrouter/google/gemini-3.1-flash-lite"   # litellm string; needs OPENROUTER_API_KEY

RUN_EXTERNAL_STORAGE = False
MINIO_OPTS = {"key": "minioadmin", "secret": "minioadmin",
              "client_kwargs": {"endpoint_url": "http://localhost:9000"}}
PG_DSN = "postgresql://postgres:pass@localhost:5432/postgres?table=crewai_records"

fmt = lambda: Formatter(FORMAT)


@tool("multiply")
def multiply(a: int, b: int) -> int:
    """Multiply two integers and return the product."""
    return a * b


def make_crew() -> Crew:
    """A fresh single-agent crew whose task requires the tool (so tool hooks fire)."""
    llm = LLM(model=MODEL)
    agent = Agent(
        role="Math Helper",
        goal="Compute arithmetic using the multiply tool.",
        backstory="You always use the multiply tool for products.",
        tools=[multiply],
        llm=llm,
    )
    task = Task(
        description="Use the multiply tool to compute 6 times 7, then report the result.",
        expected_output="A sentence stating the product.",
        agent=agent,
    )
    return Crew(agents=[agent], tasks=[task])


# 1. CrewAIAdapter — LIVE hooks capture the whole run automatically.
def method_1_live_adapter():
    adapter = CrewAIAdapter(format=FORMAT, output=str(OUT / "m1_live.jsonl"), flush_interval=0)
    try:
        make_crew().kickoff()
        adapter.flush()
    finally:
        # remove this adapter's global hooks so later methods aren't double-captured
        for unreg, fn in ((unregister_after_llm_call_hook, adapter._on_after_llm),
                          (unregister_after_tool_call_hook, adapter._on_after_tool)):
            try: unreg(fn)
            except Exception: pass


# 2. from_llm_call_context — grab the real LLM hook context during the run.
def method_2_llm_call_context():
    grabbed = []
    grab = lambda ctx: grabbed.append(ctx)            # list.append returns None (hooks must return None)
    register_after_llm_call_hook(grab)
    try:
        make_crew().kickoff()
    finally:
        unregister_after_llm_call_hook(grab)
    assert grabbed, "no after_llm_call context captured"
    rec = fmt().format_single(from_llm_call_context(grabbed[-1]))
    write_jsonl(OUT / "m2_llm_call.jsonl", [rec], mode="a")


# 3. from_tool_call_context — grab the real tool hook context (needs a tool call).
def method_3_tool_call_context():
    grabbed = []
    grab = lambda ctx: grabbed.append(ctx)
    register_after_tool_call_hook(grab)
    try:
        make_crew().kickoff()
    finally:
        unregister_after_tool_call_hook(grab)
    assert grabbed, "no after_tool_call context captured (agent didn't call the tool)"
    rec = fmt().format_single(from_tool_call_context(grabbed[-1]))
    write_jsonl(OUT / "m3_tool_call.jsonl", [rec], mode="a")


# 4. from_event — capture a real event off the CrewAI event bus.
def method_4_from_event():
    grabbed = []

    @crewai_event_bus.on(LLMCallCompletedEvent)
    def _on_llm(source, event):
        grabbed.append(event)

    make_crew().kickoff()
    assert grabbed, "no LLMCallCompletedEvent captured"
    rec = fmt().format_single(from_event(grabbed[-1]))
    write_jsonl(OUT / "m4_event.jsonl", [rec], mode="a")


# 5. from_kickoff_output — convert the real CrewOutput returned by kickoff().
def method_5_kickoff_output():
    result = make_crew().kickoff()
    rec = fmt().format_single(from_kickoff_output(result))
    write_jsonl(OUT / "m5_kickoff.jsonl", [rec], mode="a")


# 6. External storage for ONE method's REAL capture (local already proven above).
def method_6_external_storage():
    if not RUN_EXTERNAL_STORAGE:
        print("   (skipped — set RUN_EXTERNAL_STORAGE=True with MinIO/Postgres up)")
        return
    grabbed = []
    grab = lambda ctx: grabbed.append(ctx)
    register_after_llm_call_hook(grab)
    try:
        make_crew().kickoff()
    finally:
        unregister_after_llm_call_hook(grab)
    rec = fmt().format_single(from_llm_call_context(grabbed[-1]))
    write_jsonl("s3://agentscribe/crewai_llm.jsonl", [rec],
                storage_options=MINIO_OPTS, mode="a", format_name=FORMAT)
    write_jsonl(PG_DSN, [rec], format_name=FORMAT, dataset="crewai_llm")
    print("   wrote real capture to MinIO + Postgres")


if __name__ == "__main__":
    tests = [
        ("1  CrewAIAdapter (live)",     method_1_live_adapter),
        ("2  from_llm_call_context",    method_2_llm_call_context),
        ("3  from_tool_call_context",   method_3_tool_call_context),
        ("4  from_event",               method_4_from_event),
        ("5  from_kickoff_output",      method_5_kickoff_output),
        ("6  external storage (1 way)", method_6_external_storage),
    ]
    for name, fn in tests:
        try:
            fn(); print(f"[ok]   {name}")
        except Exception as exc:
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")

    print("\nLocal outputs:")
    for p in sorted(OUT.glob("*.jsonl")):
        print(f"  {p.name}: {sum(1 for _ in p.open())} record(s)")