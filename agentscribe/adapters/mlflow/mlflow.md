# AgentScribe + MLflow: Capture Guide

This guide explains how AgentScribe reads MLflow GenAI traces and converts span data into canonical fine-tuning interactions.

---

## 1. How MLflow exposes agent responses

MLflow tracing captures GenAI calls as trace objects with metadata and spans. Depending on the export path, those objects may be native MLflow trace instances or plain JSON dictionaries.

| Mechanism | Granularity | Data captured | AgentScribe entry point |
|-----------|-------------|---------------|-------------------------|
| **Trace object** | One request or agent run | Trace id, request metadata, spans | `from_trace()` |
| **Trace dictionary** | JSON export of a trace | Same data as the trace object when present | `from_trace_dict()` |
| **Span list** | Lower-level span export | Inputs, outputs, model metadata, token usage | `opentelemetry.from_spans()` via `from_trace()` |
| **CLI conversion** | Post-hoc JSON or JSONL exports | Saved traces from MLflow tracking or local files | `agentscribe convert mlflow ...` |

MLflow support is intentionally implemented on top of the OpenTelemetry/OpenInference parser because MLflow GenAI traces commonly expose span-style data.

---

## 2. Deep dive into each capture mechanism

### 2.1 Trace object - recommended for Python integrations

Use `from_trace()` when your application already has an MLflow trace object. AgentScribe looks for spans under `trace.data.spans`, then falls back to `trace.spans` or `trace.operations`.

### 2.2 Trace dictionary - recommended for JSON exports

Use `from_trace_dict()` when reading saved JSON. It is a dict-friendly alias that routes to the same parser as `from_trace()`.

### 2.3 Span parsing - shared observability path

Once spans are found, AgentScribe delegates to the OpenTelemetry adapter. Inputs and outputs become canonical messages, tool-like spans become tool call or tool response records when possible, and raw span attributes are preserved in `interaction.spans`.

---

## 3. How AgentScribe integrates with MLflow

AgentScribe uses MLflow as a post-hoc trace source:

| Mode | How it works | What you do |
|------|--------------|-------------|
| **Python conversion** | Pass a trace object or dictionary to `from_trace()`. | Format or store the returned `CanonicalInteraction`. |
| **CLI conversion** | Save trace JSON or JSONL, then convert it. | Run `agentscribe convert mlflow <file> --format <fmt> --output <path>`. |

The adapter does not require MLflow at import time. It reads trace-like objects through duck-typed accessors, which keeps AgentScribe usable in environments where MLflow is only present in production or in an export job.

---

## 4. Recommended integration: trace export conversion

```python
from agentscribe.adapters.mlflow import from_trace
from agentscribe.core.formatter import Formatter

trace = mlflow_client.get_trace("trace-id")
interaction = from_trace(trace, metadata={"dataset": "support-agent"})
record = Formatter("openai_chat").format_single(interaction)
```

The same exported trace can be converted from the command line:

```bash
agentscribe convert mlflow ./mlflow_trace.json --format sharegpt --output ./dataset.jsonl
```

---

## 5. What gets captured

AgentScribe captures the following when the trace contains it:

- User inputs and assistant outputs from span attributes.
- Tool calls and tool results when encoded in span names or attributes.
- `trace_id` from `trace.info.trace_id`, `trace_id`, or `request_id`.
- MLflow request metadata from `trace.info.request_metadata`.
- Token usage, model metadata, span ids, parent ids, and raw span attributes through the shared OpenTelemetry parser.

---

## 6. Expected JSON shape

The adapter accepts native trace objects and dictionaries. A typical JSON export looks like this:

```json
{
  "info": {
    "trace_id": "trace-123",
    "request_metadata": {"team": "support"}
  },
  "data": {
    "spans": [
      {
        "name": "llm.call",
        "attributes": {
          "input.value": "Reset my password",
          "output.value": "Here are the steps..."
        }
      }
    ]
  }
}
```

For simpler exports, a top-level `spans` or `operations` list is also accepted.