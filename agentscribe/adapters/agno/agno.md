# AgentScribe + Agno: Capture Guide

This guide explains every way Agno can record agent activity, and how AgentScribe plugs into each method to turn your agent runs into fine‑tuning data.

---

## 1. How Agno exposes agent responses

Agno offers a rich observability ecosystem with multiple capture mechanisms – from lightweight terminal debugging to production‑grade OpenTelemetry tracing persisted in your own database. The table below summarises them.

| Mechanism | Granularity | Data captured | Requires explicit setup? |
|---|---|---|---|
| **MLflow autolog** | Per agent run, model call, tool call | Prompts, completion responses, latencies, token usage, cost, tool executions, cache hits, exceptions | One call: `mlflow.agno.autolog()` |
| **AgentOS tracing** | Every span (LLM, tool, hook, retrieval, team delegation) | Agent ID, user ID, session ID, model, latency, status, tool inputs/outputs, prompt/completion tokens | `AgentOS(…, tracing=True)` |
| **Post‑hooks** | After each agent run | `RunOutput` object with `messages`, `content`, `tools`, `metrics` | Pass `post_hooks=[...]` to Agent |
| **Tool hooks** | Before/after each tool call | Tool name, arguments, result | Pass `tool_hooks=[...]` to Agent or Tool |
| **Event Streaming** | Real‑time events during execution | Run events, tool call events, hook events, reasoning steps | Access via `RunOutput.events` |
| **Debug mode** | Terminal‑printed debug logs | System prompt, user messages, tool calls | `debug_mode=True` on Agent or `AGNO_DEBUG=true` |
| **Monitoring** | Session‑level tracking on app.agno.com | Agent sessions, run history, token usage | `monitoring=True` on Agent or `AGNO_MONITOR=true` |
| **RunOutput** (return value) | After each `agent.run()` | `messages` list, `content`, `tools` (ToolExecution list), `metrics`, agent name/ID, session ID | None – always available |
| **Session messages** | Post‑hoc retrieval | All messages for a session including tool calls and system messages | Requires database (`db=...`) |
| **External providers** | Third‑party dashboards | Traces exported to Langfuse, Langsmith, Arize, etc. | Additional exporter setup |

Each method serves different scenarios, but for building a training dataset **MLflow autolog** or **post‑hooks** give the richest, most structured information with the least effort.

---

## 2. Deep dive into each capture mechanism

### 2.1 MLflow autolog – AgentScribe's primary recommended method

MLflow autolog is a **single‑line, zero‑instrumentation** capture mechanism. Calling `mlflow.agno.autolog()` automatically instruments every agent in the process, capturing comprehensive traces across model calls, tool invocations, and agent steps.

**What it captures:**
- Prompts and completion responses
- Latencies
- Metadata about different agents (function names, agent roles)
- Token usage and cost
- Cache hits
- Any exceptions raised

**Per‑trace attributes (documented by MLflow + Agno):**

| Span | Attributes captured |
|---|---|
| Agent invocation | Agent name, model, run ID, session ID |
| LLM call | Prompt tokens, completion tokens, temperature, tool calls returned |
| Tool execution | Tool name, arguments, result, duration, exception (if any) |

**Why AgentScribe prefers MLflow autolog:**
It requires the least code (one import + one function call), captures everything automatically including multi‑agent interactions, and provides structured trace data that maps directly to our canonical model. It also works with both self‑hosted and managed MLflow servers (AWS, Azure, GCP), and traces are OpenTelemetry‑native for compatibility with existing observability pipelines.

**Limitations:**
Requires installing `mlflow>=3.3` alongside Agno.

---

### 2.2 AgentOS tracing – production‑grade, database‑owned

AgentOS tracing uses OpenTelemetry‑compatible spans persisted in **your own database**. It's designed for production deployments where you need complete ownership of your trace data.

**What it captures:**

| Span type | Attributes |
|---|---|
| Run | `agent_id`, `user_id`, `session_id`, model, latency, status |
| LLM call | Model, prompt tokens, completion tokens, temperature, tool calls returned |
| Tool | Tool name, arguments, result, duration, exception |
| Pre/post hook | Hook name, duration, modified input/output |
| Retrieval | Query, vector store, k, returned docs, scores |
| Team delegation | Member name, mode, sub‑run trace |

Traces are stored in `agno_traces` and `agno_spans` tables in your database and are directly queryable via SQL.

**Why it matters for training data:**
The Agno docs explicitly state: *"Traces are your training data. Every model call, every tool result, every retrieval, every step of every run – that's the corpus you'd use to fine‑tune, evaluate, or build your next agent on."* AgentOS writes traces to your database by default – you own the schema, retention, and access patterns.

**When to use:**
When you're already using AgentOS in production and want your trace data to serve double duty as training data. However, extracting training‑ready JSONL from database spans requires additional query logic, which AgentScribe's CLI can perform.

---

### 2.3 Post‑hooks – flexible, programmable capture

Post‑hooks execute **after every agent run** and receive the full `RunOutput` object, which contains everything AgentScribe needs.

**The `RunOutput` object provides:**

| Attribute | Type | Description |
|---|---|---|
| `messages` | `Optional[List[Message]]` | All messages included in the response |
| `content` | `Optional[Any]` | The response content |
| `tools` | `Optional[List[ToolExecution]]` | All tool executions during the run |
| `metrics` | `Optional[RunMetrics]` | Token usage, cost, timing |
| `agent_id` | `Optional[str]` | ID of the agent |
| `agent_name` | `Optional[str]` | Name of the agent |
| `session_id` | `Optional[str]` | Session ID |
| `model` | `Optional[str]` | Model used |

**Example post‑hook signature:**

```python
from agno.run.run_output import RunOutput

def capture_hook(run_output: RunOutput, agent, session, run_context) -> None:
    for msg in run_output.messages:
        print(msg.role, msg.content)
```

**When to use:**
When you want fine‑grained control over what gets captured, or when you cannot add MLflow as a dependency. Post‑hooks are ideal for AgentScribe because they provide the messages list directly – mapping perfectly to our canonical model.

**Background hooks for production:**
You can mark a hook with `@hook(run_in_background=True)` so the capture logic never blocks the agent's response path:

```python
from agno.hooks import hook

@hook(run_in_background=True)
def capture_hook(run_output, agent):
    # write to storage without adding latency
```

---

### 2.4 Tool hooks – intercept individual tool calls

Tool hooks let you run custom logic before or after a specific tool is called. They receive the tool name, the function call, and its arguments, and can access the Agent or Team object.

**Example:**

```python
import time

def logger_hook(function_name: str, function_call, arguments: dict):
    start_time = time.time()
    result = function_call(**arguments)
    duration = time.time() - start_time
    print(f"Tool {function_name} took {duration:.2f}s")
    return result

agent = Agent(
    tools=[DuckDuckGoTools()],
    tool_hooks=[logger_hook],
)
```

**When to use:**
As a complement to other capture methods. Tool hooks are useful for detailed tool‑level logging but don't provide the full conversation context. AgentScribe can use them alongside post‑hooks to capture complete tool‑use trajectories.

---

### 2.5 Debug mode – quick terminal inspection

Setting `debug_mode=True` on an agent prints the system prompt, user messages, and tool calls to the terminal. It can also be enabled globally via `AGNO_DEBUG=true`.

**When to use:**
For local development and debugging only. Not suitable for production data capture because the output is unstructured terminal text.

---

### 2.6 Monitoring – cloud‑hosted session tracking

Setting `monitoring=True` on an agent (or `AGNO_MONITOR=true` globally) sends session data to Agno's cloud dashboard at `app.agno.com/sessions`.

**When to use:**
For quick visual inspection of agent runs. Not recommended for training data because the data lives on Agno's cloud, not your infrastructure.

---

### 2.7 RunOutput – always available, no setup needed

Every call to `agent.run()` returns a `RunOutput` object. It contains `messages`, `content`, `tools`, `metrics`, and metadata like `agent_name` and `session_id`.

**Example:**

```python
from agno.agent import Agent
from agno.models.openai import OpenAIChat

agent = Agent(model=OpenAIChat(id="gpt-4o"))
run_output = agent.run("Tell me a joke")

print(run_output.content)
print(run_output.messages)
print(run_output.agent_name)
print(run_output.session_id)
```

**When to use:**
For post‑hoc capture: you can iterate over the `RunOutput` after each run and extract the data you need. This works even when no hooks or tracing were configured.

---

### 2.8 Session messages – post‑hoc retrieval from database

When an agent is configured with a database (`db=...`), all messages are persisted. You can retrieve them later via:

- `session.get_messages()` – all messages including tool calls and system messages
- `agent.get_chat_history(session_id=...)` – only user and assistant messages

**When to use:**
For batch extraction of historical sessions. AgentScribe's CLI can query session messages and convert them to training data.

---

### 2.9 Event Streaming – real‑time fine‑grained events

Agno's Event Streaming System emits granular events at multiple levels during agent execution: `run_started`, `tool_call_started`, `tool_call_completed`, `reasoning_step`, `run_completed`, and many more. These events are accessible via `RunOutput.events`.

**When to use:**
For building custom real‑time monitoring dashboards. Less practical for training data due to the high volume of events.

---

## 3. How AgentScribe integrates with Agno

AgentScribe offers two integration modes:

| Mode | How it works | What you do |
|---|---|---|
| **In‑process (Python library)** | AgentScribe registers MLflow autolog or post‑hooks that fire during agent execution, building `CanonicalInteraction` objects in real time. | Add 2–3 lines of code before creating your agent. |
| **Post‑hoc (CLI)** | After the run, point AgentScribe at MLflow traces, session data, or `RunOutput` exports and convert them to a training dataset. | Run `agentscribe convert agno <source> --format <fmt> --output <path>` |

The in‑process mode via MLflow autolog is the recommended approach – it captures everything automatically, requires only one function call, and stores data in your preferred format without extra processing.

---

## 4. Recommended integration: MLflow autolog (in‑process)

**Why this is the best option:**

- ✅ Captures every agent run, model call, and tool execution automatically.
- ✅ Preserves multi‑agent interactions – agent handoffs, team delegation, member responses.
- ✅ Tracks token usage and cost for each LLM call.
- ✅ One‑line activation – no per‑agent configuration.
- ✅ Traces are OpenTelemetry‑native and can be consumed by AgentScribe's trace parser.
- ✅ Works with both self‑hosted and managed MLflow servers.
- ✅ Data is saved automatically to local storage or cloud (S3, GCS, Azure).

**What the user experience looks like:**

```python
from agentscribe.adapters.agno import AgnoAdapter

# One line to activate capture
capture = AgnoAdapter(
    format="sharegpt",                 # or "openai_chat", "alpaca", etc.
    output="s3://my-bucket/training/", # local, S3, GCS, Azure
)

agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    tools=[YFinanceTools(stock_price=True)],
)
agent.print_response("What is the stock price of Apple?")
# Training data is now in s3://my-bucket/training/
```

**What gets captured in one interaction:**

- System prompt (from agent instructions)
- User message(s)
- Assistant response(s)
- Any tool calls with arguments and returned results
- Token usage and model metadata
- Agent name, session ID

---

## 5. Alternative integration: Post‑hooks (in‑process, no MLflow dependency)

If you prefer not to add MLflow as a dependency, AgentScribe can also capture data through Agno's native post‑hooks.

**Why this is a strong alternative:**

- ✅ No additional dependencies beyond `agno`.
- ✅ Direct access to `RunOutput.messages` – the exact message list needed for training data.
- ✅ Works with any Agno agent, with or without AgentOS.
- ✅ Can be combined with tool hooks for complete tool‑use capture.
- ✅ Background execution supported via `@hook(run_in_background=True)` for zero latency impact.

**Example:**

```python
from agentscribe.adapters.agno import AgnoHookAdapter

capture = AgnoHookAdapter(format="openai_chat", output="./data.jsonl")

agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    tools=[YFinanceTools(stock_price=True)],
    post_hooks=[capture.post_hook],   # captures full messages
    tool_hooks=[capture.tool_hook],   # captures individual tool calls
)
agent.print_response("What is the stock price of Apple?")
capture.flush()  # optional – also auto‑flushed on garbage collection
```

---

## 6. What if I already use a different logging method?

| Your current setup | AgentScribe's recommendation |
|---|---|
| No logging at all | Add `AgnoAdapter` (MLflow autolog) or `AgnoHookAdapter` (post‑hooks) – one line and you're capturing. |
| Using AgentOS tracing | Keep your traces; AgentScribe's CLI can query `agno_spans` and convert them to training datasets. |
| Using MLflow autolog already | Point AgentScribe at your MLflow tracking server to extract trace data and format it. |
| Using post‑hooks already | Add AgentScribe's capture logic inside your existing hook function. |
| Using debug mode | Switch to MLflow autolog or post‑hooks for structured, production‑grade capture. |
| Using monitoring | Complement with AgentScribe for local, training‑ready data; monitoring data lives on Agno's cloud. |
| Using external providers (Langfuse, Arize, etc.) | AgentScribe's CLI can import exported traces from these platforms. |

---

## 7. Configuration examples (comprehensive)

### 7.1 MLflow autolog with local output

```python
from agentscribe.adapters.agno import AgnoAdapter
from agno.agent import Agent
from agno.models.openai import OpenAIChat

capture = AgnoAdapter(format="openai_chat", output="./agno_training.jsonl")

agent = Agent(model=OpenAIChat(id="gpt-4o"), tools=[...])
agent.print_response("Hello world")
```

### 7.2 MLflow autolog with S3 output

```python
capture = AgnoAdapter(format="sharegpt", output="s3://my-bucket/agno_data/")

# Make sure AWS credentials are configured (env vars, ~/.aws/credentials, or IAM role)
agent = Agent(...)
agent.print_response(...)
```

### 7.3 Post‑hooks with background processing

```python
from agentscribe.adapters.agno import AgnoHookAdapter
from agno.hooks import hook

capture = AgnoHookAdapter(format="alpaca", output="./data.jsonl")

@hook(run_in_background=True)
def capture_post_hook(run_output, agent, session, run_context):
    capture.post_hook(run_output, agent, session, run_context)

agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    post_hooks=[capture_post_hook],
    tool_hooks=[capture.tool_hook],
)
```

### 7.4 Post‑hoc CLI: convert MLflow traces

```bash
agentscribe convert agno-mlflow ./mlruns/ --format openai_chat --output ./dataset.jsonl
```

### 7.5 Post‑hoc CLI: convert session messages from database

```bash
agentscribe convert agno-session \
    --db-url "postgresql://user:pass@localhost:5432/agno_db" \
    --session-id "sess_123" \
    --format sharegpt \
    --output ./session_dataset.jsonl
```

---

## 8. How AgentScribe maps Agno data to canonical messages

Internally, regardless of which capture method you use, AgentScribe normalises the data into its `CanonicalInteraction` model. Here's the mapping:

| Agno concept | Canonical representation |
|---|---|
| System prompt (agent instructions) | `CanonicalMessage(role="system", content=...)` |
| User message | `CanonicalMessage(role="user", content=...)` |
| Assistant response | `CanonicalMessage(role="assistant", content=...)` |
| Tool call | `CanonicalMessage(role="tool_call", content="", tool_name=..., tool_args=...)` |
| Tool result | `CanonicalMessage(role="tool_response", content=..., tool_name=..., tool_result=...)` |
| Agent name / model | `metadata` dict |
| Run / session ID | `session_id` |
| Timestamps | Auto‑generated |

This normalised form is then converted to the desired fine‑tuning format (OpenAI chat, ShareGPT, Alpaca, etc.) by AgentScribe's Formatter.

---

## 9. Summary

- **Best for complete data with minimal code:** MLflow autolog via `AgnoAdapter`.
- **Best for zero additional dependencies:** Post‑hooks via `AgnoHookAdapter`.
- **Best for production AgentOS deployments:** AgentOS tracing + CLI batch extraction.
- **Best for logs you already have:** Session messages or `RunOutput` exports + AgentScribe CLI.
- **Always insufficient alone:** Debug mode (unstructured terminal output).

AgentScribe's Agno adapter is designed to make either the MLflow or post‑hooks path a one‑line addition, giving you a seamless, production‑grade data capture pipeline.