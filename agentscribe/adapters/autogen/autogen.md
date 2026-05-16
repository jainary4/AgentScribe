# AgentScribe + AutoGen / AG2: Capture Guide

This guide explains the AutoGen and AG2 capture surfaces AgentScribe supports and how task results, chat histories, and stream events become canonical interactions.

---

## 1. How AutoGen exposes agent responses

AutoGen and AG2 commonly expose conversation data through task results, chat history lists, and streamed AgentChat events. AgentScribe normalizes these shapes without importing AutoGen at package import time.

| Mechanism | Granularity | Data captured | AgentScribe entry point |
|-----------|-------------|---------------|-------------------------|
| **Task result** | One completed run | Messages, stop reason, token usage, optional agent metadata | `from_task_result()` |
| **Chat history** | One conversation history | User, assistant, and tool-related messages | `from_chat_history()` |
| **Stream events** | Per event from `run_stream()` | Stream spans, incremental chunks, final result messages | `from_stream_events()` |
| **Single message or event** | One AgentChat item | Canonical messages for assistant, user, tool call, or tool result payloads | `messages_from_autogen_item()` |
| **CLI conversion** | Post-hoc JSON or JSONL exports | Saved task results, chat histories, or stream event lists | `agentscribe convert autogen ...` |

Task result capture is the recommended default because it usually contains the final message sequence and usage metadata in one object.

---

## 2. Deep dive into each capture mechanism

### 2.1 Task result - recommended for datasets

Use `from_task_result()` when your application has a completed AgentChat `TaskResult` or a compatible export. AgentScribe reads `messages` or `chat_history`, converts each item into canonical messages, and stores the stop reason and token usage when present.

### 2.2 Chat history - recommended for legacy AG2 exports

Use `from_chat_history()` when you have a simple chat transcript export. AgentScribe reads `chat_history` or `messages` and normalizes them directly.

### 2.3 Stream events - recommended for streaming runs

Use `from_stream_events()` when you capture events from `AssistantAgent.run_stream()`. Event payloads are preserved as spans, tool-related events are converted into tool call and tool response messages, and final result events are promoted into the main interaction.

### 2.4 Single message normalization - fine-grained conversion

`messages_from_autogen_item()` is the lowest-level conversion surface. It can normalize assistant messages, user messages, tool call requests, tool execution results, and streaming chunks.

---

## 3. How AgentScribe integrates with AutoGen

AgentScribe offers two integration modes for AutoGen and AG2:

| Mode | How it works | What you do |
|------|--------------|-------------|
| **Python conversion** | Convert task results, histories, or stream events directly in code. | Call `from_task_result()`, `from_chat_history()`, or `from_stream_events()`. |
| **Post-hoc CLI** | Convert saved JSON exports into training records. | Run `agentscribe convert autogen <file> --format <fmt> --output <path>`. |

The CLI also accepts `ag2` as an alias.

---

## 4. Recommended integration: task result capture

```python
from agentscribe.adapters.autogen import from_task_result
from agentscribe.core.formatter import Formatter

result = agent.run(task="Draft a customer reply")
interaction = from_task_result(result, agent=agent)
record = Formatter("openai_chat").format_single(interaction)
```

The same shape can be converted from JSON with the CLI:

```bash
agentscribe convert autogen ./autogen_result.json --format sharegpt --output ./dataset.jsonl
```

---

## 5. What gets captured

AgentScribe captures the following when the data is present:

- User and assistant messages.
- Tool call requests and tool execution results.
- Stream chunks and stream events as spans.
- Stop reason and token usage from task results.
- Optional agent metadata such as name, description, and system message.

---

## 6. Expected JSON shapes

For a task result export:

```json
{
  "messages": [
    {"source": "user", "content": "Draft a reply"},
    {"source": "assistant", "content": "Here is a concise reply."}
  ],
  "stop_reason": "completed"
}
```

For a chat history export:

```json
{
  "chat_history": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi"}
  ]
}
```

For a stream export:

```json
{
  "events": [
    {"type": "TextMessage", "source": "assistant", "content": "Partial output"},
    {"messages": [{"source": "assistant", "content": "Final output"}]}
  ]
}
```
