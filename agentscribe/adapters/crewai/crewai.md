# AgentScribe + CrewAI: Capture Guide

This guide explains every way CrewAI can record agent activity, and how AgentScribe plugs into each method to turn your agent runs into fine‑tuning data.

---

## 1. How CrewAI exposes agent responses

CrewAI gives you five different mechanisms to capture what your agents do. They range from “set once and forget” to “manually inspect the result after the run”. The table below summarises them.

| Mechanism | Granularity | Data captured | Requires explicit setup? |
|-----------|-------------|---------------|--------------------------|
| **Execution Hooks** | Per LLM call / per tool call | Full message history, LLM response, tool arguments & results | One registration line |
| **Callbacks** | Per agent step or per task | Step output (`AgentAction`, `AgentFinish`, `ToolResult`) or `TaskOutput` | Yes, per agent / task |
| **Event Listeners** | Every internal event (crew start, LLM call, tool use, etc.) | Structured event objects with agent, task, output | Yes, one listener class |
| **Output Log File** | Whole run written to a file | Full agent thoughts, actions, and final answers as text or JSON | One flag on `Crew()` |
| **Kickoff Return Value** | End of run only | Final output, token usage, per‑task results | None – always available |
| **Python’s `logging` module** | Debug‑level logs throughout the run | Agent thoughts, tool calls, final answers (unstructured text) | One `import logging` block |

Each method is useful in different scenarios, but for building a training dataset **Execution Hooks** give the richest, most structured information with the least effort.

---

## 2. Deep dive into each capture mechanism

### 2.1 Execution Hooks – AgentScribe’s recommended method

Execution hooks are CrewAI’s newest interception system. They let you attach functions that run automatically **before** or **after** every LLM call and every tool call.

- **`after_llm_call`** – fires immediately after the agent receives an LLM response.  
  You receive an `LLMCallHookContext` that contains:
  - `context.messages` – the **complete conversation list** up to that point (system, user, assistant messages)
  - `context.response` – the raw LLM output string
  - `context.agent` – which agent is speaking
  - `context.task` – the task being executed
  - `context.iterations` – how many LLM calls have been made so far

- **`after_tool_call`** – fires after a tool finishes.  
  You get a `ToolCallHookContext` with:
  - `tool_name`, `tool_input`, `tool_result`
  - `agent` and `task` references

**Why AgentScribe prefers hooks:**  
They are the only mechanism that gives you the **full message history** plus the **LLM response** in one place, and they fire **automatically** on every turn. That means one registration can capture an entire multi‑step agent run, including tool usage, with zero changes to your agent logic.

---

### 2.2 Callbacks – step‑level or task‑level capture

Callbacks are passed directly to the `Agent` or `Task` constructor.

- `step_callback(step_output)` – called after each agent “step” (a thought, a tool call, or a final answer).  
  The `step_output` can be an `AgentAction`, `AgentFinish`, or `ToolResult`. You only see **what** the agent output, not the conversation context that led to it.

- `task_callback(task_output)` – called after a task completes. Receives a `TaskOutput` with `raw`, `json_dict`, `summary`, etc.

**When to use:** If you only care about final answers or monitoring steps, but **not** for building a training dataset because the missing context makes it impossible to reconstruct the prompt the model saw.

---

### 2.3 Event Listeners – passive, framework‑wide observation

Event listeners tap into CrewAI’s internal event bus. You create a class that inherits from `BaseEventListener` and register handlers for specific event types.

Relevant events include `LLMCallCompletedEvent`, `ToolUsageFinishedEvent`, and `AgentExecutionCompletedEvent`. Each event object carries agent, task, and output data.

**When to use:** When you want to capture agent activity **without modifying your agent code at all** – just adding a listener. However, the event payloads are slightly less detailed than hooks (e.g., the full message list may not be directly available), so AgentScribe treats event listeners as a secondary option.

---

### 2.4 Output log file – post‑hoc capture

Set `output_log_file=True` or `output_log_file="my_logs.json"` when creating a `Crew`. CrewAI writes a log of thoughts, actions, and final answers to that file.

- Text format (`.txt` / no extension): human‑readable but requires pattern parsing.
- JSON format (`.json`): structured, easily machine‑readable.

**When to use:** If you forgot to add hooks during the run, or you’re processing logs from a third party. AgentScribe’s CLI can convert these files to training data with `agentscribe convert crewai ./logs.json --format sharegpt --output ...`.

---

### 2.5 Kickoff return value – always available

`crew.kickoff()` returns a `CrewOutput` object with `.raw`, `.json_dict`, `.tasks_output`, and `.token_usage`. This is the final result only – no intermediate steps or message history.

**When to use:** Quick checks or extracting the final answer. Not sufficient for fine‑tuning because you lose the conversation that led to it.

---

### 2.6 Python’s built‑in logging

Enabling `logging.basicConfig(level=logging.DEBUG)` before running a crew will dump debug messages containing agent thoughts, tool calls, and outputs. The output is unstructured and requires custom parsing.

**When to use:** As a fallback if no other method was set up and you have raw debug logs. AgentScribe’s CLI can attempt to parse these, but the quality is lower.

---

## 3. How AgentScribe integrates with CrewAI

AgentScribe offers two integration modes, both using the same canonical data model:

| Mode | How it works | What you do |
|------|--------------|-------------|
| **In‑process (Python library)** | AgentScribe registers hooks or listeners that fire during agent execution, building `CanonicalInteraction` objects in real time. | Add 2–3 lines of code before creating your crew. |
| **Post‑hoc (CLI)** | After the run, point AgentScribe at a log file or `CrewOutput` export and convert it to a training dataset. | Run `agentscribe convert crewai <logfile> --format <fmt> --output <path>` |

The **in‑process mode** is the recommended approach because it captures data live, ensures you don’t miss anything, and stores it directly in the format you need (e.g., OpenAI chat, ShareGPT) without extra processing.

---

## 4. Recommended integration: Execution Hooks (in‑process)

**Why this is the best option:**

- ✅ Captures **every** LLM call with the full conversation history.
- ✅ Captures **tool calls** and their results, preserving agent tool‑use trajectories.
- ✅ One‑time registration – no per‑agent configuration.
- ✅ Works even with `verbose=False`.
- ✅ Data is saved automatically to local storage or cloud (S3, GCS, Azure).

**What the user experience looks like:**

1. Import AgentScribe’s CrewAI adapter.
2. Create a `CrewAIAdapter` with your desired output format and storage location.
3. Run your crew as usual.

> AgentScribe attaches `after_llm_call` and `after_tool_call` hooks that serialize every interaction into the canonical format and write them to your chosen storage.

**What gets captured in one interaction:**

- System prompt
- User message(s)
- Agent thoughts and final answers
- Any tool calls with arguments and returned results

All these are saved as a single, ready‑to‑use training example.

---

## 5. What if I already use a different logging method?

| Your current setup | AgentScribe’s recommendation |
|--------------------|------------------------------|
| **No logging at all** | Add the AgentScribe hook – it’s one line and doesn’t change your code. |
| **Using `step_callback` or `task_callback`** | Switch to hooks for richer data, or keep both. |
| **Using `output_log_file`** | Either add the hook for live capture, or use the CLI to convert the log file after the run. |
| **Using Event Listeners** | You can keep your listener and also add hooks – they coexist peacefully. |
| **Using MLflow autolog** | AgentScribe can also read MLflow traces via its CLI or Agno adapter. |

---

## 6. Configuration examples

*In‑process (hooks)*

```python
from agentscribe.adapters.crewai import CrewAIAdapter

# One line to activate capture
capture = CrewAIAdapter(
    format="sharegpt",            # or "openai_chat", "alpaca", etc.
    output="s3://my-bucket/training/",  # local, S3, GCS, Azure
)

crew = Crew(agents=[...], tasks=[...])
crew.kickoff()
capture.flush()
```
# Training data is now in s3://my-bucket/training/

*Post-hoc converter*

```python
from agentscribe.adapters.crewai import from_llm_call_context

interaction = from_llm_call_context(saved_hook_context)
```

```bash
agentscribe convert crewai ./run_logs.json --format openai_chat --output ./dataset.jsonl
```
