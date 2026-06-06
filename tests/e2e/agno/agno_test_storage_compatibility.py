from agno.agent import Agent
from agno.models.openrouter import OpenRouter
from agentscribe.adapters.agno import from_run_output
from agentscribe.core.formatter import Formatter
from agentscribe.storage import write_jsonl

# One table for both agents; the `dataset` column separates them.
DSN = "postgresql://postgres:pass@localhost:5432/postgres?table=agent_records"

def agent():
    return Agent(model=OpenRouter(id="google/gemini-3.1-flash-lite"))

AGENTS = {
    "agent1": (agent(), ["Benefits of solar energy?", "One drawback?", "Translate 'good morning' to French."]),
    "agent2": (agent(), ["Tell me a Python joke.", "Explain list comprehension in one line.", "What is a decorator?", "Name one PEP."]),
}

for name, (ag, prompts) in AGENTS.items():
    for p in prompts:
        record = Formatter("openai_chat").format_single(from_run_output(ag.run(p)))
        write_jsonl(DSN, [record], format_name="openai_chat", dataset=name)   # ← library INSERTs a row

# Verify accumulation + separation (read_jsonl doesn't apply to the row-store, so query directly).
import psycopg
with psycopg.connect("postgresql://postgres:pass@localhost:5432/postgres") as conn:
    for name, _ in AGENTS.items():
        n = conn.execute("SELECT count(*) FROM agent_records WHERE dataset = %s", (name,)).fetchone()[0]
        print(name, "->", n, "rows")