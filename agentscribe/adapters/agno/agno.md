# AgentScribe + Agno: Capture Guide

This guide explains the Agno capture surfaces AgentScribe supports and how run outputs, sessions, traces, and event streams become dataset-ready interactions.

---

## 1. How Agno exposes agent responses

Agno can expose agent work through run outputs, sessions, event streams, metrics, and trace exports. AgentScribe keeps the adapter duck-typed so these shapes can be parsed without importing Agno at package import time.

| Mechanism | Granularity | Data captured | AgentScribe entry point |
|-----------|-------------|---------------|-------------------------|
| **Run output** | One Agent, Team, or Workflow run | Messages or prompt/output pair, metrics, tools, agent metadata | `from_run_output()` |
| **Session export** | Multiple runs in one session | One interaction per run, shared session id | `from_session()` |
| **Trace export** | One traced run | Spans, trace id, messages inferred from span attributes | `from_trace()` |
| **Event stream** | Per emitted event | Event spans, message deltas or content | `from_event_stream()` |
| **Hook-style collector** | In-process post-run hook | Recorded run outputs | `AgnoTraceCollector` |

Run output capture is the recommended default for training data because it usually contains the final prompt, response, message history, and usage metrics in one place.

---

## 2. Deep dive into each capture mechanism

### 2.1 Run output - recommended for datasets

`from_run_output()` accepts Agent, Team, and Workflow run result shapes. When message history is available under `messages`, `chat_history`, or `history`, AgentScribe normalizes that list directly. Otherwise, it falls back to an input/output pair using fields such as `input`, `prompt`, `message`, `content`, `output`, `response`, or `raw`.

### 2.2 Session export - batch conversion

`from_session()` reads a session export and converts each run into a separate `CanonicalInteraction`. The session id is copied onto each interaction when it is available.

### 2.3 Trace export - observability conversion

`from_trace()` reads Agno span exports and delegates span parsing to the OpenTelemetry adapter. This is useful when Agno traces are already being collected through AgentOS or OpenInference-compatible tracing.

### 2.4 Event stream - live or logged events

`from_event_stream()` preserves each event as a span and promotes message-like payloads into canonical assistant messages. This is useful for stream logs and incremental run records.

### 2.5 Hook-style collector - in-process capture

`AgnoTraceCollector` provides `record_run_output()` and `post_hook()` helpers so application code can record run results after an agent completes.

---

## 3. How AgentScribe integrates with Agno

AgentScribe offers two integration modes for Agno:

| Mode | How it works | What you do |
|------|--------------|-------------|
| **In-process collector** | Record each run output after an Agent, Team, or Workflow completes. | Use `AgnoTraceCollector` or call `from_run_output()` directly. |
| **Post-hoc CLI** | Convert saved run, session, or trace exports into formatted training data. | Run `agentscribe convert agno <file> --format <fmt> --output <path>`. |

Both modes produce `CanonicalInteraction` objects with `source_framework="agno"`, message lists, provenance metadata, optional token usage, tools, and trace details.

---

## 4. Recommended integration: run output capture

```python
from agentscribe.adapters.agno import from_run_output
from agentscribe.core.formatter import Formatter

run_output = agent.run("Create a concise onboarding checklist.")
interaction = from_run_output(run_output, agent=agent)
record = Formatter("openai_chat").format_single(interaction)
```

The same shape can be converted from JSON with the CLI:

```bash
agentscribe convert agno ./agno_run.json --format openai_chat --output ./dataset.jsonl
```

---

## 5. What gets captured

AgentScribe captures the following when the data is present:

- Message history, prompt/output pairs, or streamed assistant content.
- Agent name, id, model, instructions, and tools.
- Run id, trace id, and session id.
- Metrics, usage, and token usage.
- Tool call metadata from run outputs.
- Raw event and trace spans for observability context.

---

## 6. Expected JSON shapes

For a single run output:

```json
{
  "run_id": "run-1",
  "session_id": "session-1",
  "messages": [
    {"role": "user", "content": "Plan my launch checklist"},
    {"role": "assistant", "content": "Start with scope, dates, and owners."}
  ],
  "metrics": {"input_tokens": 24, "output_tokens": 18}
}
```

For a session export:

```json
{
  "session_id": "session-1",
  "runs": [
    {"input": "Draft a reply", "output": "Here is a concise reply..."}
  ]
}
```

For a trace export, pass a top-level `spans` list or a `data.spans` object compatible with OpenTelemetry/OpenInference attributes.