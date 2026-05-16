# AgentScribe + LangGraph: Capture Guide

This guide explains the capture surfaces LangGraph exposes and how AgentScribe turns graph states and stream events into fine-tuning records.

---

## 1. How LangGraph exposes agent responses

LangGraph stores agent progress in graph state and can emit structured stream events while a graph runs. AgentScribe supports both paths without importing LangGraph at package import time.

| Mechanism | Granularity | Data captured | AgentScribe entry point |
|-----------|-------------|---------------|-------------------------|
| **State snapshots** | End of invoke, checkpoint, or exported state | Messages, thread id, graph metadata, extra state fields | `from_state()` |
| **Stream events** | Per graph update or streamed token/message event | Updates, values, tasks, debug records, optional message chunks | `from_stream_events()` |
| **Recording wrapper** | Around `invoke()` and `stream()` calls | The returned state or stream output from the compiled graph | `LangGraphRecorder` / `wrap_graph()` |
| **CLI conversion** | Post-hoc JSON or JSONL exports | Saved states or stream event lists | `agentscribe convert langgraph ...` |

State snapshots are the recommended capture surface when you want clean training examples, because they usually contain the final `messages` list for the thread.

---

## 2. Deep dive into each capture mechanism

### 2.1 State snapshots - recommended for datasets

Most LangGraph chat workflows keep conversation history under a `messages`, `chat_history`, or `conversation` key. AgentScribe reads those fields, normalizes each framework message into `CanonicalMessage`, and stores remaining state fields under `interaction.extra["state"]`.

Use this when you have the output from `graph.invoke()`, a checkpoint value, or a JSON export of a graph state.

### 2.2 Stream events - recommended for observability exports

LangGraph streams can produce tuples or structured stream parts. AgentScribe normalizes both forms into spans so the raw event trail is preserved. Events in `updates`, `values`, `debug`, and `tasks` modes are scanned for messages. `messages` mode chunks are kept as spans by default and can be promoted into dataset messages when `include_message_chunks=True`.

Use this when your application already records stream output or when you want to preserve graph execution details alongside the dataset-ready messages.

### 2.3 Recording wrapper - in-process capture

`LangGraphRecorder` wraps a compiled graph and records the result of `invoke()` and `stream()` calls into an `InteractionCollector`. The original graph is not mutated, so the wrapper can be introduced around existing graph objects.

Use this when you want live capture in application code without changing the graph definition.

---

## 3. How AgentScribe integrates with LangGraph

AgentScribe offers two integration modes for LangGraph:

| Mode | How it works | What you do |
|------|--------------|-------------|
| **In-process wrapper** | Wrap the compiled graph, run it normally, and record each invoke or stream result. | Call `wrap_graph(graph)` or instantiate `LangGraphRecorder`. |
| **Post-hoc CLI** | Convert saved state snapshots or stream event exports into formatted training data. | Run `agentscribe convert langgraph <file> --format <fmt> --output <path>`. |

Both paths produce `CanonicalInteraction` objects with `source_framework="langgraph"`, optional `session_id`, `thread_id`, graph metadata, spans, and normalized messages.

---

## 4. Recommended integration: state capture

State capture is the best default for fine-tuning data because it preserves the final conversation without duplicating transient stream chunks.

```python
from agentscribe.adapters.langgraph import from_state
from agentscribe.core.formatter import Formatter

state = graph.invoke(
    {"messages": [{"role": "user", "content": "Summarize this ticket."}]},
    config={"configurable": {"thread_id": "support-123"}},
)

interaction = from_state(
    state,
    config={"configurable": {"thread_id": "support-123"}},
    graph=graph,
)

record = Formatter("openai_chat").format_single(interaction)
```

The same shape can be converted from JSON with the CLI:

```bash
agentscribe convert langgraph ./langgraph_state.json --format openai_chat --output ./dataset.jsonl
```

---

## 5. What gets captured

AgentScribe captures the following when the data is present:

- System, user, assistant, tool call, and tool response messages.
- `thread_id` or checkpoint id from LangGraph config.
- Graph name and graph class metadata.
- Stream events as spans, including updates, values, tasks, debug records, and optional message chunks.
- Non-message state fields under `interaction.extra["state"]`.

---

## 6. Expected JSON shapes

For state exports, pass either the state object directly or a wrapper containing `state`, `config`, and optional `metadata`:

```json
{
  "state": {
    "messages": [
      {"role": "user", "content": "Hello"},
      {"role": "assistant", "content": "Hi"}
    ]
  },
  "config": {"configurable": {"thread_id": "thread-1"}}
}
```

For stream exports, pass `events` or `stream`:

```json
{
  "events": [
    ["updates", {"messages": [{"role": "assistant", "content": "Done"}]}]
  ],
  "config": {"configurable": {"thread_id": "thread-1"}}
}
```