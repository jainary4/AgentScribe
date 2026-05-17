# AgentScribe + OpenInference: Capture Guide

This guide explains how AgentScribe handles OpenInference traces through the shared OpenTelemetry parser.

---

## 1. How OpenInference exposes agent responses

OpenInference commonly represents LLM and tool activity as OpenTelemetry-compatible spans with GenAI semantic conventions. AgentScribe treats OpenInference as a compatibility layer on top of the OpenTelemetry adapter.

| Mechanism | Granularity | Data captured | AgentScribe entry point |
|-----------|-------------|---------------|-------------------------|
| **Span list** | One trace or run | Messages, tool calls, token usage, model/provider metadata | `from_spans()` |
| **Trace object** | One exported trace | Span list plus trace id | `from_trace()` |
| **CLI conversion** | Post-hoc JSON or JSONL exports | Saved traces from OpenInference-compatible collectors | `agentscribe convert openinference ...` |
| **Compatibility layer** | Python package alias | Reuses the OpenTelemetry parser for OpenInference shapes | `agentscribe.adapters.openinference` |

The underlying parsing logic is shared with OpenTelemetry because OpenInference exports usually follow the same span-oriented model with additional semantic conventions.

---

## 2. Deep dive into each capture mechanism

### 2.1 Trace conversion - recommended for datasets

Use `from_trace()` when you have an OpenInference trace export. AgentScribe reads the span list and promotes prompt, completion, and tool activity into canonical messages.

### 2.2 Span conversion - recommended for collector exports

Use `from_spans()` when your export already consists of a span list. This is common when traces are persisted directly from a collector or observability backend.

### 2.3 Compatibility package - simplest import path

The `agentscribe.adapters.openinference` package wraps the OpenTelemetry parsing helpers so OpenInference-specific code can import a dedicated adapter package without duplicating the parser implementation. Direct Python imports preserve `openinference` as the source label by default.

---

## 3. How AgentScribe integrates with OpenInference

AgentScribe uses OpenInference as a post-hoc trace source:

| Mode | How it works | What you do |
|------|--------------|-------------|
| **Python conversion** | Convert OpenInference trace objects or span lists directly in code. | Call `from_trace()` or `from_spans()`. |
| **Post-hoc CLI** | Convert saved OpenInference JSON or JSONL exports. | Run `agentscribe convert openinference <file> --format <fmt> --output <path>`. |

When using either the Python API or the CLI, AgentScribe preserves `openinference` as the source label in the resulting canonical interaction metadata.

---

## 4. Recommended integration: trace export conversion

```python
from agentscribe.adapters.openinference import from_trace
from agentscribe.core.formatter import Formatter

trace = {
    "trace_id": "trace-123",
    "spans": [
        {
            "name": "llm.call",
            "attributes": {
                "openinference.span.kind": "LLM",
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
agentscribe convert openinference ./openinference_trace.json --format sharegpt --output ./dataset.jsonl
```

---

## 5. What gets captured

AgentScribe captures the following when the trace contains it:

- Prompt and completion messages.
- Tool calls and tool responses derived from tool-related span attributes.
- Trace id and span id metadata.
- Raw span attributes and span events.
- Token usage, model name, provider metadata, and OpenInference span kind markers.

---

## 6. Expected JSON shape

```json
{
  "trace_id": "trace-123",
  "spans": [
    {
      "name": "llm.call",
      "attributes": {
        "openinference.span.kind": "LLM",
        "input.value": "Reset my password",
        "output.value": "Here are the steps...",
        "gen_ai.request.model": "gpt-4.1"
      }
    }
  ]
}
```
