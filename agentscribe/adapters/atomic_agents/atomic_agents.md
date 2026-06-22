# Capturing Atomic Agents with AgentScribe

A guide to every surface AgentScribe exposes for turning **Atomic Agents** runs into
fine-tuning datasets — what each does, how it works, when to use it, its limitations,
and a paste-and-run example for each.

---

## Versions

These examples were written and verified against:

| Package | Version |
|---|---|
| `atomic-agents` | **2.8.1** |
| `instructor` | 1.14.5 |
| `pydantic` | 2.13.4 |
| `agentscribe` | current `main` |

```bash
pip install "atomic-agents==2.8.1" instructor openai
pip install -e .            # agentscribe, from the repo root
export OPENROUTER_API_KEY=sk-...     # or use any instructor-compatible client
```

---

## 1. How Atomic Agents is structured — and why it changes how we capture tools

Atomic Agents is **atomic and composable**: an agent is nothing more than

```
input schema  ->  LLM call  ->  output schema
```

An agent **decides**; it does not **act**. This one design choice is the key to
everything below.

### Tools are *external* to the agent

In frameworks like **CrewAI** or **Agno**, the framework runs tools *inside* the agent
loop: the agent calls a tool, the framework executes it, captures the
`(tool_name, args, result)` triple, and feeds it back — all internally. A hook can read
that structured tool call straight from the framework.

In **Atomic Agents there is no such loop.** The agent's *output schema* may contain a
field like `tool="calculator"` and `tool_input="2+2"` — but that is just a **decision
expressed as data**. Atomic never runs the calculator. **You** run it, and **you** feed
the result back in as the next input. The framework therefore never holds a structured
tool call; the only place that information exists is in your code.

> **Consequence:** for tool data, *the developer is the source of truth*, not the
> framework. This is not a limitation of AgentScribe — it is intrinsic to Atomic's
> design.

### Why "chat completion" and "tool completion" are captured differently

Because of the above, AgentScribe offers **two styles** of capture:

| Style | What a tool run looks like | Produces `tool_calls`? | Surfaces |
|---|---|---|---|
| **Conversational** | the tool result is injected as a normal turn; the decision is assistant text. It reads as a multi-turn chat. | **No** | `from_agent_response`, `from_agent_run`, `from_chat_history`, `AtomicAgentsAdapter`, `record_response`, `record_history` |
| **Structured** | proper `user → assistant{tool_calls} → tool → assistant` | **Yes** | `from_tool_interaction`, `record_tool_interaction` |

- If your agent **doesn't use tools** (a plain chat/structured agent), use any
  conversational surface. They capture exactly what happened: user → assistant.
- If your agent **uses tools** and you want real `tool_calls` data (for OpenAI /
  ShareGPT fine-tuning), use the **structured** surface and hand it the
  `(name, args, result)` you executed. The conversational surfaces *can* capture a tool
  run too, but only as flattened chat (the tool result becomes a turn) — fine for
  text/Alpaca formats, lossy for `tool_calls`.

The one-line rule: **chat → any conversational surface; tool calls you want as
`tool_calls` → `from_tool_interaction` (or `record_tool_interaction`), because you ran
the tool.**

### The pipeline

Every surface ends up in the same two-step pipeline:

```
capture surface  ->  CanonicalInteraction  ->  Formatter(format)  ->  write_jsonl(target)
                     (neutral representation)   (openai_chat, ...)     (file / s3 / gs / postgres)
```

---

## 2. Three ways to capture — and why two of them are production-grade

All three tiers can end up persisting data, so the real difference is **not** "writes vs
doesn't." It's about **where the capture is triggered** and, crucially, **where the
format / target / write logic lives**:

- the `from_*` functions push that logic to **every call site**;
- the collector and adapter **centralize** it in one place.

That centralization is exactly what makes the collector and adapter production-grade. The
clearest way to see it is to capture the same three runs three different ways.

### Way 1 — `from_*` functions: convert only, so you repeat the boilerplate everywhere

A `from_*` function only builds a `CanonicalInteraction`. It does **not** format and does
**not** write — *you* do, at every site:

```python
from agentscribe.adapters.atomic_agents import from_agent_response

q1 = ChatInput(chat_message="Greet the user.")
r1 = chat_agent.run(q1)
write_jsonl("data.jsonl",
    [Formatter("openai_chat").format_single(from_agent_response(r1, prompt=q1, agent=chat_agent))],
    mode="a")

q2 = ChatInput(chat_message="What's our refund window?")
r2 = chat_agent.run(q2)
write_jsonl("data.jsonl",
    [Formatter("openai_chat").format_single(from_agent_response(r2, prompt=q2, agent=chat_agent))],
    mode="a")

q3 = ChatInput(chat_message="Summarize the conversation.")
r3 = chat_agent.run(q3)
write_jsonl("data.jsonl",
    [Formatter("openai_chat").format_single(from_agent_response(r3, prompt=q3, agent=chat_agent))],
    mode="a")
```

**The limitation, concretely:** the format (`"openai_chat"`), the target
(`"data.jsonl"`), and the whole convert → format → write sequence are **duplicated at
every capture site**.

- Want ShareGPT instead of OpenAI? Edit it in **three** places.
- Want to write to Postgres instead of a file? **Three** places.
- Forget `mode="a"` at one site and that write **truncates** the file.
- Three separate writes = three file opens (or three DB round-trips).

This is fine for a one-off script. It does **not** scale to a real codebase where agents
run in many places.

### Way 2 — `AtomicAgentsTraceCollector`: declare once, batch the write

```python
from agentscribe.adapters.atomic_agents import AtomicAgentsTraceCollector

# format + target are declared ONCE, here:
collector = AtomicAgentsTraceCollector(format_name="openai_chat", output_path="data.jsonl")

for text in ["Greet the user.", "What's our refund window?", "Summarize the conversation."]:
    q = ChatInput(chat_message=text)
    collector.record_response(chat_agent.run(q), prompt=q, agent=chat_agent)  # no format, no write

collector.flush()   # ONE write for the whole batch
```

**Why this is production-grade:**

- Format and target live in **one** place — switching to `sharegpt` or a
  `postgres://…` URI is a **one-line** change, and every record stays consistent.
- Each capture site is a single `record_*` call with **no** formatting and **no** I/O.
- The whole batch is persisted in **one** operation — one file open, or one DB
  transaction — instead of N. For 500 runs that's 1 write instead of 500.

### Way 3 — `AtomicAgentsAdapter`: instrument once, capture automatically

```python
from agentscribe.adapters.atomic_agents import AtomicAgentsAdapter

cap = AtomicAgentsAdapter(format="openai_chat", output="data.jsonl")  # format + target: ONCE
cap.attach(chat_agent)                                                # instrument ONCE

for text in ["Greet the user.", "What's our refund window?", "Summarize the conversation."]:
    chat_agent.run(ChatInput(chat_message=text))
    cap.capture(chat_agent)        # on 2.8.1: snapshot after run (see hook-timing note in section 4)

cap.flush()   # also fires automatically at process exit (atexit)
```

**Why this is production-grade:** like the collector, format + target are declared once
and the write is batched — **plus** an `atexit` safety flush, so you don't lose data if
you forget to flush or the process exits. The design intent is fully hands-off (attach
once, runs captured by a hook with **no** per-site call); on 2.8.1 you add one
`capture()` after each run because the hook fires before the assistant turn lands in
history (section 4). Even so, your run sites carry **no** formatting or I/O.

### The takeaway

| | format + target defined | per-site code | writes |
|---|---|---|---|
| `from_*` functions | **at every site** | convert + format + write | **N writes** |
| TraceCollector | **once** (constructor) | one `record_*` call | **1 batched write** |
| Adapter | **once** (constructor) | `run` (+ `capture` on 2.8.1) | **1 batched write + atexit** |

`from_*` is the right tool for one-off captures or per-record control (a different
file/format per interaction). The collector and adapter are what you reach for in a real
system, because the format, the target, and the write logic stop being copy-pasted
across your codebase and live in exactly one place.

### The three axes, summarized

| | capture triggered by | writes by itself? | write happens |
|---|---|---|---|
| `from_*` functions | **you, per run** | **no** — you add `write_jsonl` | per call (your choice) |
| `AtomicAgentsTraceCollector` | you, per run (`record_*`) | yes (`flush`) | **once per batch** |
| `AtomicAgentsAdapter` | **a hook, automatically** | yes (`flush`) | once per batch / atexit |

> "One adapter (or collector) per agent" is only about **file separation** (one dataset
> per agent) — it's orthogonal to auto-vs-batch. Want everything in one file? Use one
> adapter and attach all agents to it.

---

## 3. Shared setup (preamble for every example)

All examples below assume this preamble has run. It builds one chat agent and one
tool agent and a demo tool executor.

```python
import os
from typing import Literal, Optional

import instructor
from openai import OpenAI
from pydantic import Field

from atomic_agents import AtomicAgent, AgentConfig, BaseIOSchema
from atomic_agents.context import SystemPromptGenerator

from agentscribe.core.formatter import Formatter
from agentscribe.storage import write_jsonl

client = instructor.from_openai(OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"]))
MODEL = "google/gemini-2.5-flash"


# ---- a plain chat agent (no tools) ----
class ChatInput(BaseIOSchema):
    """User message."""
    chat_message: str = Field(..., description="The user's message.")

class ChatOutput(BaseIOSchema):
    """Assistant reply."""
    chat_message: str = Field(..., description="The assistant's reply.")

chat_agent = AtomicAgent[ChatInput, ChatOutput](AgentConfig(
    client=client, model=MODEL,
    system_prompt_generator=SystemPromptGenerator(
        background=["You are a concise, friendly assistant."])))


# ---- a tool-deciding agent ----
class ToolInput(BaseIOSchema):
    """User message."""
    chat_message: str = Field(..., description="The user's message.")

class ToolOutput(BaseIOSchema):
    """A tool call OR a final answer."""
    tool: Optional[Literal["calculator", "search"]] = Field(None, description="Tool to call.")
    tool_input: Optional[str] = Field(None, description="Input for the tool.")
    final_answer: Optional[str] = Field(None, description="Final answer once you have enough info.")

tool_agent = AtomicAgent[ToolInput, ToolOutput](AgentConfig(
    client=client, model=MODEL,
    system_prompt_generator=SystemPromptGenerator(
        background=["You answer questions, using the calculator tool for arithmetic."],
        output_instructions=["Set 'tool' and 'tool_input' to call a tool; set 'final_answer' when done."])))


def execute_tool(name: str, inp: str) -> str:
    """YOU run the tool — Atomic only decides. (demo only; never eval untrusted input)"""
    if name == "calculator":
        return str(eval(inp))
    return f"(search) top result for {inp!r}"
```

---

## 4. Method reference

### `from_agent_response` — the workhorse (conversational)

**What it does.** Turns a single `agent.run(...)` response into a clean interaction.
Works in two modes.

**How it works.** It pulls the natural-language text out of the response schema (looking
at fields like `chat_message`, `content`, `final_answer`, …). With no `history`, it
emits a single `user → assistant` pair. With `history=agent.history`, it captures the
whole thread and appends the final response if it isn't already the last turn. The full
structured response object is also kept in `metadata["extra"]["structured_output"]`.

**When to use.** A plain chat/structured agent: one run, one answer. The simplest
capture there is.

**Limitations.** Conversational only — if a *tool-deciding* agent returns
`tool="calculator"`, that decision is captured as **text**, not as `tool_calls`. For
structured tool data use `from_tool_interaction`. Like all `from_*` functions, it does
not write — you call `Formatter` + `write_jsonl` yourself (see section 2).

**This method serves two scenarios — both shown:**

```python
from agentscribe.adapters.atomic_agents import from_agent_response

# Scenario A: single prompt -> response
q = ChatInput(chat_message="Give me one fun fact about the moon.")
reply = chat_agent.run(q)
interaction = from_agent_response(reply, prompt=q, agent=chat_agent)
write_jsonl("chat_single.jsonl", [Formatter("openai_chat").format_single(interaction)], mode="w")

# Scenario B: capture the whole multi-turn thread by passing history
chat_agent.run(ChatInput(chat_message="Hi!"))
reply2 = chat_agent.run(ChatInput(chat_message="Now explain photosynthesis in one line."))
interaction2 = from_agent_response(reply2, history=chat_agent.history, agent=chat_agent)
write_jsonl("chat_thread.jsonl", [Formatter("openai_chat").format_single(interaction2)], mode="w")
```

---

### `from_agent_run` — convenience alias for an I/O pair (conversational)

**What it does.** Identical behavior to `from_agent_response`, but its signature takes
the **input schema and output schema** explicitly: `from_agent_run(input, output, ...)`.

**How it works.** It simply calls `from_agent_response(output, prompt=input, ...)`.

**When to use.** When you're holding both the input and the output objects and want code
that reads as "here is my I/O pair." Purely a naming/ergonomics choice for structured
agents.

**Limitations.** Same as `from_agent_response` — conversational, no `tool_calls`, no
write of its own.

```python
from agentscribe.adapters.atomic_agents import from_agent_run

q = ChatInput(chat_message="Define entropy in one sentence.")
reply = chat_agent.run(q)
interaction = from_agent_run(q, reply, agent=chat_agent)   # (input, output)
write_jsonl("agent_run.jsonl", [Formatter("alpaca").format_single(interaction)], mode="w")
```

---

### `from_chat_history` — capture a whole conversation (conversational)

**What it does.** Normalizes an Atomic `ChatHistory` (or a raw message list) into a clean
multi-turn conversation.

**How it works.** It reads each message's role and text from `history.get_history()`
and rebuilds them as canonical messages, in order.

**When to use.** You've already built up a back-and-forth on one agent and want the
entire thread as one record.

**Limitations.** Conversational only. In particular, tool results you injected with
`agent.add_tool_result(...)` come back under their **stored role** (user/system) —
because Atomic has no dedicated tool role — so a tool flow captured this way looks like
plain chat, with no `tool_calls`.

```python
from agentscribe.adapters.atomic_agents import from_chat_history

chat_agent.run(ChatInput(chat_message="Hi!"))
chat_agent.run(ChatInput(chat_message="What's a quark?"))
chat_agent.run(ChatInput(chat_message="And a lepton?"))

interaction = from_chat_history(chat_agent.history, agent=chat_agent)
write_jsonl("history.jsonl", [Formatter("sharegpt").format_single(interaction)], mode="w")
```

---

### `from_tool_interaction` — the only path to real `tool_calls` (structured)

**What it does.** Builds one structured tool-use interaction
(`user → assistant{tool_calls} → tool → assistant`) from the pieces you pass.

**How it works.** You hand it the `tool_calls` you executed — a list of
`{"name", "args", "result"}`. For each, it emits a paired tool-call + tool-response
message with a shared `tool_call_id`. It also attaches a root-level `tools` JSON schema
array: pass `tool_schemas=[...]` for high-quality schemas, or omit it and AgentScribe
auto-derives a minimal valid one from your call args (so the line still passes OpenAI
fine-tuning validation). Works for **any number** of tool calls.

**When to use.** A tool-using agent where you want `tool_calls` fine-tuning data. Prefer
it even for Alpaca/text output — a declared tool chain is cleaner than a history where
the tool result was flattened into a turn.

**Limitations.** You must declare the tools yourself (Atomic won't give them to you), and
it builds the interaction *only* from what you pass — it doesn't read agent history, so
intermediate reasoning turns aren't included unless you put them in.

```python
from agentscribe.adapters.atomic_agents import from_tool_interaction

# your normal Atomic tool flow: decide -> YOU run the tool -> feed result back -> answer
q = ToolInput(chat_message="what is 50 + 70?")
decision = tool_agent.run(q)                                  # agent DECIDES
result = execute_tool(decision.tool, decision.tool_input)     # YOU execute
final = tool_agent.run(ToolInput(chat_message=f"[{decision.tool}] result = {result}")).final_answer

interaction = from_tool_interaction(
    q.chat_message,
    tool_calls=[{"name": decision.tool, "args": {"input": decision.tool_input}, "result": result}],
    final_answer=final,
    system="You answer questions, using the calculator tool for arithmetic.",
    agent=tool_agent,
    # tool_schemas=[{...}]   # optional; auto-derived if omitted
)
write_jsonl("tool.openai.jsonl", [Formatter("openai_chat").format_single(interaction)], mode="w")
```

One capture → many formats (the same interaction emits `tool_calls` for `openai_chat`,
`function_call`/`observation` for `sharegpt`, and clean text for `alpaca`):

```python
for fmt in ["openai_chat", "sharegpt", "alpaca", "prompt_completion"]:
    write_jsonl(f"tool.{fmt}.jsonl", [Formatter(fmt).format_single(interaction)], mode="w")
```

---

### `from_log_event` — fold in monitoring/hook payloads (observability)

**What it does.** Normalizes a logging/monitoring/hook event emitted around a run into an
interaction, recording the event as a span.

**How it works.** It reads the event's `type`/`event_type`, and extracts
`input`/`prompt`/`input_schema` and `output`/`response`/`output_schema` if present,
adding them as user/assistant turns; the raw event is preserved as a span.

**When to use.** You already have an observability/telemetry layer emitting event objects
and want those folded into the dataset. A niche surface — not your primary capture path.

**Limitations.** Only as rich as the event payload, and produces no `tool_calls`.

```python
from agentscribe.adapters.atomic_agents import from_log_event

event = {
    "type": "agent_run",
    "input": ChatInput(chat_message="Hello"),
    "output": ChatOutput(chat_message="Hi! How can I help?"),
}
interaction = from_log_event(event)
write_jsonl("log_event.jsonl", [Formatter("prompt_completion").format_single(interaction)], mode="w")
```

---

### `AtomicAgentsTraceCollector` — batch many runs into one file

An in-memory collector. You `record_*` interactions and `flush()` writes them all at
once. Its `record_response`/`record_history` are **conversational**;
`record_tool_interaction` is **structured**. See section 2 for why this is
production-grade (format + target declared once, one batched write).

**When to use.** Collecting many runs to one target; choosing the output format at write
time. One collector per agent/dataset (separation must happen at capture time).

**Limitations.** Held in memory until `flush()`, with no `atexit` safety net (unlike the
adapter) — flush yourself. `flush()` defaults to **`append=True`**: each flush appends
its records and **drains** what it wrote, so re-flushing within a process does **not**
duplicate. Pass `append=False` to overwrite the file each flush instead. (Because append
persists across separate process runs, delete the file for a clean start, or use
`append=False`.)

#### Example 1 — collector **without** `record_tool_interaction` (conversational batch)

For plain chat agents. Uses `record_response` (and `record_history` is available too).

```python
from agentscribe.adapters.atomic_agents import AtomicAgentsTraceCollector

collector = AtomicAgentsTraceCollector(format_name="sharegpt", output_path="chat_batch.jsonl")
for text in ["One fact about Mars?", "One fact about Venus?", "One fact about Mercury?"]:
    q = ChatInput(chat_message=text)
    collector.record_response(chat_agent.run(q), prompt=q, agent=chat_agent)
collector.flush()                 # appends by default (idempotent); pass append=False to overwrite
```

#### Example 2 — collector **with** `record_tool_interaction` (structured batch)

For tool-using agents where you want `tool_calls`. You run the tool loop, accumulate the
calls, and record one structured interaction per task.

```python
from agentscribe.adapters.atomic_agents import AtomicAgentsTraceCollector

def run_tool_loop(agent, user_text, *, max_steps=6):
    """decide -> run tool -> feed back -> repeat (with a guard against infinite loops)."""
    tool_calls, final, current = [], None, ToolInput(chat_message=user_text)
    for _ in range(max_steps):
        d = agent.run(current)
        if d.tool is not None:
            res = execute_tool(d.tool, d.tool_input)
            tool_calls.append({"name": d.tool, "args": {"input": d.tool_input}, "result": res})
            current = ToolInput(chat_message=f"[{d.tool}] result = {res}")
        else:
            final = d.final_answer
            break
    return tool_calls, final

collector = AtomicAgentsTraceCollector(format_name="openai_chat", output_path="tool_batch.jsonl")
for text in ["what is 50 + 70?", "what is (12 * 3) - 4?"]:
    calls, final = run_tool_loop(tool_agent, text)
    collector.record_tool_interaction(text, tool_calls=calls, final_answer=final, agent=tool_agent)
collector.flush()
```

---

### `AtomicAgentsAdapter` — hands-off auto-capture

**What it does.** `attach(agent)` registers Atomic's `completion:response` hook so runs
are captured automatically; `flush()` writes. It inherits buffering, an `atexit`
safe-flush, and context-manager support from `BaseAdapter`, and always writes in
**append** mode. See section 2 for why this is production-grade.

**How it works.** On each completion it snapshots the agent's history (keyed by
`id(agent)`, so the latest snapshot per agent wins). Attaching several agents to one
adapter writes them all to that adapter's single file — one interaction **per agent**
(its whole thread), not per run.

**When to use.** Turnkey capture of chat/structured agents — "attach once and forget."
One adapter per agent if you want separate files.

**Limitations (important).**
1. **Conversational only** — the hook fires on LLM completions, which happen *before*
   you execute a tool, so the adapter never sees tool calls.
2. **Hook timing.** On atomic-agents 2.8.1, `completion:response` fires *inside*
   `client...create()` — **before** the assistant message is appended to history. So a
   snapshot taken on the hook captures the **user turn only**. To capture the assistant
   turn, read history *after* `run()` returns. The reliable pattern is to call
   `capture(agent)` after each run (it reads the now-complete history):

```python
from agentscribe.adapters.atomic_agents import AtomicAgentsAdapter

cap = AtomicAgentsAdapter(format="openai_chat", output="auto_chat.jsonl")
cap.attach(chat_agent)                                  # registers the hook
chat_agent.run(ChatInput(chat_message="Tell me about the moon."))
cap.capture(chat_agent)                                 # snapshot AFTER run -> full user+assistant
cap.flush()
```

> Note: the adapter appends and never overwrites, so a stale output file accumulates
> across runs. Delete the file (or use a fresh path) when you want a clean capture.

---

## 5. Choosing a surface

| You have… | Surface | Style | `tool_calls`? |
|---|---|---|---|
| one chat/structured run | `from_agent_response` | conversational | no |
| both input + output objects | `from_agent_run` | conversational | no |
| a full multi-turn thread | `from_chat_history` | conversational | no |
| a tool chain you executed | `from_tool_interaction` | **structured** | **yes** |
| logging/monitoring events | `from_log_event` | event/span | no |
| many chat runs to batch | `AtomicAgentsTraceCollector.record_response` | conversational | no |
| many tool runs to batch | `AtomicAgentsTraceCollector.record_tool_interaction` | **structured** | **yes** |
| every run, hands-off | `AtomicAgentsAdapter.attach` + `capture` + `flush` | conversational | no |

---

## 6. Output formats and storage targets

**Formats** (pick at write time with `Formatter(fmt)` or the collector's `format_name`):
`openai_chat`, `sharegpt`, `alpaca`, `prompt_completion`, `preference`. The same
captured interaction can be emitted to several.

**Targets** — `write_jsonl` / the collector's `output_path` accept:

- a local path: `./data.jsonl`
- a cloud URI: `s3://bucket/data.jsonl`, `gs://…`, `az://…`
- a **Postgres** JSONB row store URI (records are inserted as rows)

So "write to a file" and "store in a database" are the same call with a different target —
and with the collector/adapter, switching between them is a one-line change in one place.

---

## 7. Summary

Atomic Agents does not execute tools — your code does. So:

- **Chat / structured agents** → any conversational surface. `AtomicAgentsAdapter` is the
  turnkey option (remember to `capture()` after `run()` so the assistant turn is
  included on 2.8.1).
- **Tool-calling agents, and you want real `tool_calls`** → `from_tool_interaction` /
  `record_tool_interaction`. You pass the `{name, args, result}` you already have. This
  is the only path to structured tool calls, by design: in Atomic, the developer is the
  source of truth for tool execution.
- **One-off vs production** → `from_*` is great for one-offs and per-record control, but
  it couples format + target + write to every site. For a real system, the collector and
  adapter centralize all of that in one place and batch the write — that is what makes
  them production-grade.