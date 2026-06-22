
"""
Minimal AgentScribe capture for a tool-calling Atomic agent.
 
The three AgentScribe lines are marked  # <-- AGENTSCRIBE.
Everything else is your normal Atomic Agents workflow.
"""
 
import os
from typing import Literal, Optional
 
import instructor
from openai import OpenAI
from pydantic import Field
 
from atomic_agents import AtomicAgent, AgentConfig, BaseIOSchema
from atomic_agents.context import SystemPromptGenerator
 
from agentscribe.adapters.atomic_agents import AtomicAgentsTraceCollector  # <-- AGENTSCRIBE (import)
 
 
# ---- your normal Atomic setup -----------------------------------------------
client = instructor.from_openai(OpenAI(
    base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"]))
 
 
class In(BaseIOSchema):
    """User message."""
    chat_message: str = Field(..., description="The user's message.")
 
 
class Out(BaseIOSchema):
    """A tool call OR a final answer."""
    tool: Optional[Literal["calculator"]] = Field(None, description="Tool to call.")
    tool_input: Optional[str] = Field(None, description="Input for the tool.")
    final_answer: Optional[str] = Field(None, description="Final answer.")
 
 
SYSTEM = "You are a math assistant. Use the calculator tool for arithmetic."
 
agent = AtomicAgent[In, Out](AgentConfig(
    client=client, model="google/gemini-3.1-flash-lite",
    system_prompt_generator=SystemPromptGenerator(
        background=[SYSTEM],
        output_instructions=[
            "Break multi-step math into separate calculator calls.",
            "Set 'tool'/'tool_input' to call the tool; when done set 'final_answer'.",
        ])))
 
 
def execute_tool(name: str, inp: str) -> str:
    """YOU run the tool — Atomic only decides. (demo only; don't eval untrusted input)"""
    return str(eval(inp)) if name == "calculator" else f"(no tool '{name}')"
 
 
# ---- one collector for the whole run ----------------------------------------
collector = AtomicAgentsTraceCollector(                                          # <-- AGENTSCRIBE (create)
    format_name="openai_chat", output_path="out.jsonl")
 
 
def run(user_input: In, *, max_steps: int = 6) -> None:
    """Decide -> run tool -> feed result back -> repeat, then capture the chain."""
    tool_calls, final_answer, current = [], None, user_input
    for _ in range(max_steps):                      # step cap = no infinite loop
        decision = agent.run(current)               # Atomic: decide only
        if decision.tool is not None:
            result = execute_tool(decision.tool, decision.tool_input)
            tool_calls.append({
                "name": decision.tool,
                "args": {"input": decision.tool_input},
                "result": result,
            })
            current = type(user_input)(chat_message=f"[{decision.tool}] result = {result}")
        else:
            final_answer = decision.final_answer
            break
    else:
        final_answer = final_answer or "(stopped: max tool steps reached)"
 
    collector.record_tool_interaction(                                           # <-- AGENTSCRIBE (capture)
        user_input.chat_message,
        tool_calls=tool_calls,       # any number of calls; schemas auto-derived
        final_answer=final_answer,
        system=SYSTEM,
        agent=agent)
 
 
if __name__ == "__main__":
    for i in range(30):
        run(In(chat_message="what is (25 + 35) * 2 / 3?"))
        run(In(chat_message="what is 144 / 12 + 8?"))    # call run() as many times as you like
                                                                # <-- AGENTSCRIBE (write)
    collector.flush()

    print("wrote out.jsonl")