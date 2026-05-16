# AgentScribe + Atomic Agents: Capture Guide

This guide explains the Atomic Agents capture surfaces AgentScribe supports and how chat histories, run responses, and log-like events become dataset-ready interactions.

---

## 1. How Atomic Agents exposes agent responses

Atomic Agents can expose useful capture data through message history objects, final run responses, and hook or monitoring payloads emitted around an agent run. AgentScribe keeps the adapter duck-typed so these shapes can be parsed without importing Atomic Agents at package import time.

| Mechanism | Granularity | Data captured | AgentScribe entry point |
|-----------|-------------|---------------|-------------------------|
| **Chat history** | One conversation history | Messages plus optional agent metadata | `from_chat_history()` |
| **Run response** | One agent run | Prompt, final response, optional structured output, agent metadata | `from_agent_response()` |
| **Input/output pair** | One complete run | Input schema, output schema, optional history | `from_agent_run()` |
| **Hook or log event** | Per emitted event | Event spans, prompt/input, output/response | `from_log_event()` |
| **Collector** | In-process capture | Recorded responses and events | `AtomicAgentsTraceCollector` |
| **CLI conversion** | Post-hoc JSON or JSONL exports | Saved history, response, or event payloads | `agentscribe convert atomic ...` |

Run response capture is the recommended default for training data because it preserves the prompt, the assistant output, and any structured response payload in one place.

---

## 2. Deep dive into each capture mechanism

### 2.1 Chat history - recommended when message lists already exist

Use `from_chat_history()` when your application already has a `ChatHistory`-style object or a simple list of messages. AgentScribe reads from `messages` or `history`, normalizes the message list, and attaches agent metadata when an agent object is provided.

### 2.2 Run response - recommended for datasets

Use `from_agent_response()` when you have the final output from `agent.run()` or `agent.run_async()`. If you also have the original prompt or full history, AgentScribe includes those so the final interaction is more useful for fine-tuning.

Structured response objects are preserved in `interaction.extra["structured_output"]` when they can be converted into a mapping.

### 2.3 Input/output pair - explicit run capture

Use `from_agent_run()` when your application stores the input and output schemas separately. This is a convenience path that routes to `from_agent_response()`.

### 2.4 Hook or log event - recommended for observability payloads

Use `from_log_event()` when Atomic Agents emits monitoring, hook, or logging payloads around execution. AgentScribe preserves the raw event as a span and promotes any input/output fields into canonical user and assistant messages.

### 2.5 Collector - in-process recording

`AtomicAgentsTraceCollector` is a lightweight in-memory collector for wrapper-style integrations. It can record full responses with `record_response()` and event payloads with `on_log_event()`.

---

## 3. How AgentScribe integrates with Atomic Agents

AgentScribe offers two integration modes for Atomic Agents:

| Mode | How it works | What you do |
|------|--------------|-------------|
| **In-process collector** | Record each run response or emitted event in Python. | Use `AtomicAgentsTraceCollector` or call `from_agent_response()` directly. |
| **Post-hoc CLI** | Convert saved JSON exports of responses, histories, or events. | Run `agentscribe convert atomic <file> --format <fmt> --output <path>`. |

The CLI accepts both `atomic` and `atomic_agents` as source names.

---

## 4. Recommended integration: run response capture

```python
from agentscribe.adapters.atomic_agents import from_agent_response
from agentscribe.core.formatter import Formatter

response = agent.run("Summarize the support issue.")
interaction = from_agent_response(
    response,
    prompt="Summarize the support issue.",
    agent=agent,
)
record = Formatter("openai_chat").format_single(interaction)
```

The same shape can be converted from JSON with the CLI:

```bash
agentscribe convert atomic ./atomic_run.json --format openai_chat --output ./dataset.jsonl
```

---

## 5. What gets captured

AgentScribe captures the following when the data is present:

- Message history from `messages` or `history`.
- Final assistant response from a run output.
- Structured response payloads under `interaction.extra["structured_output"]`.
- Agent metadata including class, system prompt, model, schemas, tools, and context providers.
- Hook or monitoring payloads as spans.

---

## 6. Expected JSON shapes

For a run response export:

```json
{
  "prompt": "Summarize the support issue.",
  "response": {"summary": "The user needs a password reset."}
}
```

For a chat history export:

```json
{
  "messages": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi"}
  ]
}
```

For an event export:

```json
{
  "event_type": "agent.completed",
  "input": "Summarize the issue.",
  "output": "The user needs a password reset."
}
```
