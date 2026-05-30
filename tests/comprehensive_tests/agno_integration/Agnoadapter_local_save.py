from pathlib import Path
from agno.agent import Agent
from agno.models.openrouter import OpenRouter
from agno.tools.duckduckgo import DuckDuckGoTools
from agentscribe.adapters.agno import from_run_output
from agentscribe.storage import write_jsonl
from  agentscribe.core.formatter import Formatter

# 1. Define the agent (no hooks)
model = OpenRouter(id="google/gemini-3.1-flash-lite")
agent = Agent(model=model, tools=[DuckDuckGoTools()])

# 2. Run and capture manually
print("Running agent…")
run1 = agent.run("What is the current weather in Toronto?")
run2 = agent.run("Tell me a joke about Python.")

# 3. Convert to canonical interactions
interaction1 = from_run_output(run1)
interaction2 = from_run_output(run2)

# 4. Format and write to JSONL
formatter = Formatter(format="openai_chat")
records = [
    formatter.format_single(interaction1),
    formatter.format_single(interaction2),
]
write_jsonl("./agno_training.jsonl", records)

# 5. Inspect
output_path = Path("agno_training.jsonl")
if output_path.exists():
    print("\nGenerated training data (first 3 lines):")
    for i, line in enumerate(output_path.read_text().splitlines()[:3], 1):
        print(f"Record {i}: {line[:120]}...")
else:
    print("No output file generated – check console for errors.")