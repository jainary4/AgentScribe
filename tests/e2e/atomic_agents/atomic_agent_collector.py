"""Comprehensive test: 3-agent pipeline, each with multiple tool calls, captured separately."""

import os
import json
from pathlib import Path
from typing import Literal, Optional
from pydantic import Field
import instructor
from openai import OpenAI

from atomic_agents import AtomicAgent, AgentConfig, BaseIOSchema
from atomic_agents.context import SystemPromptGenerator

from agentscribe.adapters.atomic_agents import from_tool_interaction
from agentscribe.core.formatter import Formatter
from agentscribe.storage import write_jsonl


# -----------------------------------------------------------------------------
# 1. Setup: client, output directory, and format
# -----------------------------------------------------------------------------
OUT_DIR = Path("./out_multi_agent")
OUT_DIR.mkdir(exist_ok=True)

client = instructor.from_openai(
    OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
)

# We'll produce OpenAI‑chat (tool_calls) format for each agent.
FORMAT = "prompt_completion"


# -----------------------------------------------------------------------------
# 2. Tool execution stubs (replace with real tools in production)
# -----------------------------------------------------------------------------
def execute_tool(tool_name: str, tool_input: str) -> str:
    """Simulate tool execution. In reality, call your APIs/DBs."""
    if tool_name == "calculator":
        try:
            # Evaluate safely – only for demo!
            return str(eval(tool_input))
        except Exception as e:
            return f"Error: {e}"
    elif tool_name == "search":
        return f"Search results for '{tool_input}': [mock result 1, mock result 2]"
    elif tool_name == "database_query":
        return f"DB query '{tool_input}' returned: [id=1, value=42]"
    else:
        return f"Unknown tool: {tool_name}"


# -----------------------------------------------------------------------------
# 3. Agent 1: Calculator agent (2‑3 math tools)
# -----------------------------------------------------------------------------
class CalcInput(BaseIOSchema):
    """Input schema for the calculator agent."""
    chat_message: str = Field(..., description="User's math question.")

class CalcOutput(BaseIOSchema):
    """Output schema for the calculator agent."""
    tool: Optional[Literal["calculator"]] = Field(None, description="Tool to call.")
    tool_input: Optional[str] = Field(None, description="Expression to evaluate.")
    final_answer: Optional[str] = Field(None, description="Final numeric answer.")

calc_agent = AtomicAgent[CalcInput, CalcOutput](
    AgentConfig(
        client=client,
        model="google/gemini-2.5-flash",
        system_prompt_generator=SystemPromptGenerator(
            background=["You are a math assistant. Use the calculator tool for arithmetic."],
            output_instructions=[
                "If a calculation is needed, set 'tool' to 'calculator' and 'tool_input' to the expression.",
                "Once you have the result, provide 'final_answer' with the numeric answer."
            ]
        )
    )
)


# -----------------------------------------------------------------------------
# 4. Agent 2: Search agent (2‑3 search queries)
# -----------------------------------------------------------------------------
class SearchInput(BaseIOSchema):
    """Input schema for the search agent."""
    chat_message: str = Field(..., description="The information to search for.")

class SearchOutput(BaseIOSchema):
    """output schema for the search agent."""
    tool: Optional[Literal["search"]] = Field(None, description="Tool to call.")
    tool_input: Optional[str] = Field(None, description="Search query.")
    final_answer: Optional[str] = Field(None, description="Summarised search result.")

search_agent = AtomicAgent[SearchInput, SearchOutput](
    AgentConfig(
        client=client,
        model="google/gemini-2.5-flash",
        system_prompt_generator=SystemPromptGenerator(
            background=["You are a research assistant. Use the search tool to gather facts."],
            output_instructions=[
                "When you need more info, set 'tool' to 'search' and 'tool_input' to the query.",
                "After receiving results, set 'final_answer' with a concise summary."
            ]
        )
    )
)


# -----------------------------------------------------------------------------
# 5. Agent 3: Database agent (2‑3 queries) + final summariser
# -----------------------------------------------------------------------------
class DBInput(BaseIOSchema):
    """Input schema for the database agent."""
    chat_message: str = Field(..., description="Database query or context.")

class DBOutput(BaseIOSchema):
    """output schema for the database agent."""
    tool: Optional[Literal["database_query"]] = Field(None, description="Tool to call.")
    tool_input: Optional[str] = Field(None, description="SQL-like query.")
    final_answer: Optional[str] = Field(None, description="Final answer after DB lookup.")

db_agent = AtomicAgent[DBInput, DBOutput](
    AgentConfig(
        client=client,
        model="google/gemini-2.5-flash",
        system_prompt_generator=SystemPromptGenerator(
            background=["You are a data analyst. Query the database to answer questions."],
            output_instructions=[
                "If you need data, set 'tool' to 'database_query' and 'tool_input' to the query.",
                "Once you have the data, provide 'final_answer' with the conclusion."
            ]
        )
    )
)


# -----------------------------------------------------------------------------
# 6. Helper to run an agent with multiple tool calls and capture
# -----------------------------------------------------------------------------
def run_agent_with_tools(
    agent: AtomicAgent,
    user_input: BaseIOSchema,
    agent_name: str,
    system_prompt: str,
) -> dict:
    """
    Run an agent that may make multiple tool decisions.
    Returns a dict with:
      - 'interaction': CanonicalInteraction ready for formatting
      - 'final_answer': the final answer string
    """
    tool_calls = []
    final_answer = None

    # We'll loop until the agent provides a final answer (no tool).
    # In a real workflow, you might have more complex logic.
    current_input = user_input
    while True:
        decision = agent.run(current_input)

        if decision.tool is not None:
            # Execute the tool
            result = execute_tool(decision.tool, decision.tool_input)
            tool_calls.append({
                "name": decision.tool,
                "args": {"input": decision.tool_input},
                "result": result,
            })
            # Feed the result back as a new user message (Atomic style)
            current_input = type(user_input)(
                chat_message=f"[{decision.tool}] result = {result}"
            )
            # Continue to let the agent decide next step
        else:
            # No tool requested – this is the final answer
            final_answer = decision.final_answer
            break

    # Build the canonical interaction with all tool calls and the final answer
    interaction = from_tool_interaction(
        user_message=user_input.chat_message,
        tool_calls=tool_calls,
        final_answer=final_answer,
        system=system_prompt,
        agent=agent,
        metadata={"agent_name": agent_name},
    )
    return {"interaction": interaction, "final_answer": final_answer}


# -----------------------------------------------------------------------------
# 7. Define the pipeline flow (3 agents in sequence)
# -----------------------------------------------------------------------------
def main():
    # 7.1 Agent 1: calculate something complex (2‑3 steps)
    print("=== Agent 1: Calculator ===")
    q1 = CalcInput(chat_message="Compute (25 + 35) * 2 and then divide by 3.")
    result1 = run_agent_with_tools(calc_agent, q1, "calculator", 
                                   "You are a math assistant. Use the calculator tool for arithmetic.")
    print("  Final answer:", result1["final_answer"])

    # 7.2 Agent 2: search for info about the result (2‑3 searches)
    print("\n=== Agent 2: Search ===")
    q2 = SearchInput(chat_message=f"Find interesting facts about the number {result1['final_answer']}.")
    result2 = run_agent_with_tools(search_agent, q2, "search",
                                   "You are a research assistant. Use the search tool to gather facts.")
    print("  Final answer:", result2["final_answer"])

    # 7.3 Agent 3: query a database with the search summary (2‑3 queries)
    print("\n=== Agent 3: Database ===")
    q3 = DBInput(chat_message=f"Based on '{result2['final_answer']}', query the database for related records.")
    result3 = run_agent_with_tools(db_agent, q3, "database",
                                   "You are a data analyst. Query the database to answer questions.")
    print("  Final answer:", result3["final_answer"])

    # -------------------------------------------------------------------------
    # 8. Write each agent's interaction to a separate JSONL file (one format)
    # -------------------------------------------------------------------------
    for agent_name, interaction_data in [
        ("calc", result1["interaction"]),
        ("search", result2["interaction"]),
        ("db", result3["interaction"]),
    ]:
        # Format the single interaction
        formatter = Formatter(FORMAT)
        record = formatter.format_single(interaction_data)
        # Write to a separate file
        out_file = OUT_DIR / f"{agent_name}.{FORMAT}.jsonl"
        write_jsonl(out_file, [record], mode="a")
        print(f"  Wrote {agent_name} to {out_file}")

    # Optionally, show how to output to multiple formats for a single agent
    # e.g., for the first agent, produce all 4 formats:
    for fmt in ["prompt_completion"]:
        formatter = Formatter(fmt)
        record = formatter.format_single(result1["interaction"])
        write_jsonl(OUT_DIR / f"calc_multi.{fmt}.jsonl", [record], mode="a")
    print("\nAll formats written for the calculator agent.")


if __name__ == "__main__":
    main()