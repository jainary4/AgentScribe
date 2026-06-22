"""
AgentScribe + Atomic Agents — minimal, real-agent example for the docs.
 
Two capture surfaces, two agents each, ONE file per agent (separation happens at
capture time — one adapter / one collector per agent, never one shared file):
 
  PART A  AtomicAgentsAdapter   auto, conversational   2 chat agents, run once each
  PART B  AtomicAgentsTraceCollector  manual, structured tool calls
                                      2 agents, each calling 2 different tools
 
Run:  export OPENROUTER_API_KEY=...   &&   python example_atomic_capture.py
"""
 
import os
from pathlib import Path
from typing import Literal, Optional
 
import instructor
from openai import OpenAI
from pydantic import Field
 
from atomic_agents import AtomicAgent, AgentConfig, BaseIOSchema
from atomic_agents.context import SystemPromptGenerator
 
from agentscribe.adapters.atomic_agents import AtomicAgentsAdapter, AtomicAgentsTraceCollector
 
OUT = Path("./out_atomic"); OUT.mkdir(exist_ok=True)
client = instructor.from_openai(OpenAI(
    base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"]))
MODEL = "google/gemini-3.1-flash-lite"
 
 
def agent(in_t, out_t, background, instructions=None):
    return AtomicAgent[in_t, out_t](AgentConfig(
        client=client, model=MODEL,
        system_prompt_generator=SystemPromptGenerator(
            background=[background], output_instructions=instructions or [])))
 
 
# =========================================================================== #
# PART A — Adapter (auto, conversational): one adapter per agent -> one file each
# =========================================================================== #
class ChatIn(BaseIOSchema):
    """User message."""
    chat_message: str = Field(..., description="The user's message.")
 
class ChatOut(BaseIOSchema):
    """Assistant reply."""
    chat_message: str = Field(..., description="The assistant's reply.")
 
 
def run_adapter_demo():
    print("PART A — adapter (one per agent)")
    poet = agent(ChatIn, ChatOut, "You are a poet. Answer in a single playful couplet.")
    chef = agent(ChatIn, ChatOut, "You are a chef. Answer with one concise cooking tip.")
 
    for name, a, q in [("poet", poet, "Tell me about the moon."),
                       ("chef", chef, "How do I keep pasta from sticking?")]:
        cap = AtomicAgentsAdapter(format="alpaca", output=str(OUT / f"adapter_{name}.jsonl"))
        #cap.attach(a)                 # hooks completion:response (auto-capture)
        a.run(ChatIn(chat_message=q)) # the run fires the hook -> snapshot
        cap.capture(a)
        cap.flush()                   # one agent's thread -> its own file
        print(f"  wrote adapter_{name}.jsonl")
 
 
# =========================================================================== #
# PART B — Trace collector (structured): each agent calls 2 different tools
# =========================================================================== #
class ToolIn(BaseIOSchema):
    """User message."""
    chat_message: str = Field(..., description="The user's message.")
 
class ToolOut(BaseIOSchema):
    """A tool call OR a final answer."""
    tool: Optional[Literal["calculator", "search", "database_query"]] = Field(None, description="Tool to call.")
    tool_input: Optional[str] = Field(None, description="Input for the tool.")
    final_answer: Optional[str] = Field(None, description="Final answer once done.")
 
 
def execute_tool(name: str, inp: str) -> str:
    """YOU run the tool — Atomic only decides. (demo stubs)"""
    if name == "calculator":
        return str(eval(inp))                                  # demo only
    if name == "search":
        return f"(search) top result for {inp!r}"
    if name == "database_query":
        return f"(db) rows for {inp!r}: [id=1, value=42]"
    return f"(unknown tool {name})"
 
 
def run_tool_loop(a, user_text: str, *, max_steps: int = 6):
    """Decide -> run tool -> feed result back -> repeat. Returns (tool_calls, final)."""
    tool_calls, final, current = [], None, ToolIn(chat_message=user_text)
    for _ in range(max_steps):
        d = a.run(current)
        if d.tool is not None:
            result = execute_tool(d.tool, d.tool_input)
            tool_calls.append({"name": d.tool, "args": {"input": d.tool_input}, "result": result})
            current = ToolIn(chat_message=f"[{d.tool}] result = {result}")
        else:
            final = d.final_answer
            break
    return tool_calls, final
 
 
def run_collector_demo():
    print("PART B — trace collector (one per agent, structured tool calls)")
    analyst = agent(
        ToolIn, ToolOut,
        "You are an analyst with a calculator and a search tool.",
        ["Use 'calculator' for math and 'search' for facts; set tool/tool_input to call, final_answer when done."])
    ops = agent(
        ToolIn, ToolOut,
        "You are an ops assistant with a database_query tool and a search tool.",
        ["Use 'database_query' to look up records and 'search' for external info; set final_answer when done."])
 
    jobs = [
        ("analyst", analyst, "What is 15% of 240, then find one fact about that number?"),  # calculator + search
        ("ops", ops, "Look up plan 'pro' in the database, then search for its competitor pricing."),  # db + search
    ]
    for name, a, q in jobs:
        col = AtomicAgentsTraceCollector(format_name="alpaca", output_path=str(OUT / f"collector_{name}.jsonl"))
        tool_calls, final = run_tool_loop(a, q)
        col.record_tool_interaction(q, tool_calls=tool_calls, final_answer=final, agent=a)  # schemas auto-derived
        col.flush()                  # one agent's tool interaction -> its own file
        print(f"  wrote collector_{name}.jsonl  ({len(tool_calls)} tool calls)")
 
 
if __name__ == "__main__":
    run_adapter_demo()
    run_collector_demo()
    print(f"\nfiles in {OUT}/")