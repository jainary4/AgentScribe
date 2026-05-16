# AgentScribe + OpenTelemetry: Capture Guide

This guide explains how AgentScribe reads OpenTelemetry span exports and converts them into canonical interactions for dataset generation.

---

## 1. How OpenTelemetry exposes agent responses

OpenTelemetry records execution as spans with attributes, events, and ids. AgentScribe reads trace-like exports and interprets GenAI and tool-related attributes as dataset messages.

| Mechanism | Granularity | Data captured | AgentScribe entry point |
|-----------|-------------|---------------|-------------------------|
| **Span list** | One trace or run | Messages, tool calls, token usage, model/provider metadata | `from_spans()` |
| **Trace object** | One exported trace | Span list plus trace id | `from_trace()` |
| **Single span parsing** | One span | Prompt/completion messages and raw attributes | `messages_from_span()` / `span_attributes()` |
| **CLI conversion** | Post-hoc JSON or JSONL exports | Saved traces from OTEL collectors or span stores | `agentscribe convert opentelemetry ...` |

Span-list conversion is the recommended default because OTEL collectors and exporters commonly emit arrays of spans with attribute-rich payloads.

---

## 2. Deep dive into each capture mechanism

### 2.1 Span list - recommended for datasets

Use `from_spans()` when you already have a list of spans from an OTEL exporter, collector, or stored JSON document. AgentScribe scans span attributes for input and output messages, token usage, provider metadata, and tool call or tool result fields.

### 2.2 Trace object - recommended for JSON exports

Use `from_trace()` when your export wraps spans inside a trace object. AgentScribe reads `spans` or `children`, then preserves the trace id when present.

### 2.3 Single span parsing - low-level helpers

Use `messages_from_span()` when you only need prompt and completion messages from a single span. `span_attributes()` returns a merged attribute dictionary from dict-like or SDK-like span objects.

---

## 3. How AgentScribe integrates with OpenTelemetry

AgentScribe uses OpenTelemetry as a post-hoc trace source:

| Mode | How it works | What you do |
|------|--------------|-------------|
| **Python conversion** | Convert spans or trace objects directly in code. | Call `from_spans()` or `from_trace()`. |
| **Post-hoc CLI** | Convert saved OTEL JSON or JSONL exports into training records. | Run `agentscribe convert opentelemetry <file> --format <fmt> --output <path>`. |

This adapter is designed to work with plain mappings and exported trace payloads, so the OpenTelemetry SDK does not need to be importable when parsing saved files.

---

## 4. Recommended integration: span export conversion

```python
from agentscribe.adapters.opentelemetry import from_trace
from agentscribe.core.formatter import Formatter

trace = {
    "trace_id": "trace-123",
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

interaction = from_trace(trace)
record = Formatter("openai_chat").format_single(interaction)
```

The same export can be converted from the command line:

```bash
agentscribe convert opentelemetry ./otel_trace.json --format openai_chat --output ./dataset.jsonl
```

---

## 5. What gets captured

AgentScribe captures the following when the span data contains it:

- Prompt and completion messages from flattened or structured span attributes.
- Tool call and tool response messages from tool-related span attributes.
- Trace id and span id metadata.
- Raw span attributes and span events under `interaction.spans`.
- Token usage, model name, and provider metadata.

---

## 6. Expected JSON shapes

For a trace export:

```json
{
  "trace_id": "trace-123",
  "spans": [
    {
      "name": "llm.call",
      "attributes": {
        "input.value": "Reset my password",
        "output.value": "Here are the steps...",
        "gen_ai.usage.total_tokens": 42
      }
    }
  ]
}
```

For a flattened message export:

```json
{
  "spans": [
    {
      "attributes": {
        "llm.input_messages.0.message.role": "user",
        "llm.input_messages.0.message.content": "Hello",
        "llm.output_messages.0.message.role": "assistant",
        "llm.output_messages.0.message.content": "Hi"
      }
    }
  ]
}
```
