# AgentScribe

**Capture every agent conversation — tool calls, reasoning, multi‑turn threads — and convert them into production‑ready fine‑tuning datasets.**

AgentScribe is a cross‑framework Python library and CLI that records LLM interactions from the most popular agentic AI frameworks, normalises them into a canonical data model, and exports the results in the exact formats required by fine‑tuning platforms (OpenAI chat, ShareGPT, Alpaca, etc.). No hidden services, no vendor lock‑in — your traces stay on your own storage.

---

## What is AgentScribe?

When you run AI agents in production, every conversation is valuable training data. But each framework formats its logs differently, and none of them output data that's ready for fine‑tuning. AgentScribe fixes this.

- 📡 **Hooks into your agent** with a single line of code (or converts existing logs post‑hoc).
- 🧬 **Normalises everything** into a single, framework‑agnostic conversation structure.
- 🧵 **Exports to any training format** — OpenAI chat, ShareGPT (with function calls), Alpaca instruction‑tuning, prompt‑completion, or DPO preference pairs.
- 💾 **Writes directly to your storage** — local disk, S3, GCS, Azure Blob. Your data never leaves your infrastructure.

AgentScribe is built for AI service companies, applied AI developers, and anyone who runs agentic workflows and wants to continuously improve their models.

---

## Features

- **One‑line capture** – Add a middleware, callback, or hook to your agent and start recording immediately.
- **Canonical data model** – All interactions (user messages, assistant responses, tool calls, tool results, multi‑turn threads) are represented consistently regardless of framework.
- **Multi‑format export** – Built‑in formatters for:
  - OpenAI Chat (`{"messages": [{"role": ..., "content": ...}]}`)
  - ShareGPT (with `function_call` and `observation` roles for tool‑use)
  - Alpaca (`instruction`, `input`, `output`, `history`)
  - Prompt‑Completion (legacy `prompt`/`completion`)
  - Preference pairs (chosen/rejected for DPO/RLHF)
- **Multi‑framework support** – Works with LangGraph, CrewAI, Agno, AutoGen, Atomic Agents, and raw observability exports (AgentOps, MLflow).
- **Cloud storage first** – Output directly to S3, GCS, or Azure Blob without intermediate services.
- **CLI for post‑hoc conversion** – Convert existing log files or exported traces from AgentOps or MLflow into dataset files.
- **Open‑source (MIT)** – No API keys, no quotas, no telemetry.

---

## Supported Frameworks & Platforms

AgentScribe provides native adapters for the following:

| Framework | Integration Mechanism |
|---|---|
| **LangGraph** | Middleware |
| **CrewAI** | After‑LLM‑Call Hooks |
| **Agno** | MLflow autolog hooks |
| **AutoGen (AG2)** | Runtime logging parser |
| **Atomic Agents / Atoms SDK** | Loguru sink |

External observability platforms can be used as data sources via the CLI:

| Platform | Mode |
|---|---|
| **AgentOps** | REST API pull → canonical model → formatted dataset |
| **MLflow** | Trace parsing from local or remote tracking servers |

---

## Storage Backends

AgentScribe writes data directly to the storage of your choice. The output path prefix determines the backend:

| Prefix | Storage Backend |
|---|---|
| `./` or `/home/...` | Local filesystem |
| `s3://bucket/path/` | Amazon S3 |
| `gs://bucket/path/` | Google Cloud Storage |
| `az://container/path/` | Azure Blob Storage |

No additional infrastructure is required; the library streams formatted JSONL directly to the target location.

---

## Architecture & File Tree

```
agentscribe/
├── agentscribe/                  # Main Python package
│   ├── __init__.py               # Package initialiser, exposes quick API
│   ├── core/
│   │   ├── __init__.py
│   │   ├── canonical.py          # CanonicalInteraction and CanonicalMessage dataclasses
│   │   └── formatter.py          # Format converters (OpenAI, Alpaca, ShareGPT, etc.)
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── langgraph.py          # LangGraph middleware adapter
│   │   ├── crewai.py             # CrewAI hooks adapter
│   │   ├── agno.py               # Agno/MLflow adapter
│   │   ├── autogen.py            # AutoGen runtime log adapter
│   │   ├── atomic_agents.py      # Atomic Agents / Atoms SDK Loguru adapter
│   │   └── agentops.py           # AgentOps API adapter (Mode B)
│   ├── storage.py                # Multi‑backend storage writer (local, S3, GCS, Azure)
│   └── cli.py                    # CLI entry point (Click/Typer)
├── pyproject.toml                # Build configuration and dependencies
├── LICENSE                       # MIT License
└── README.md                     # This file
```

### Core files explained

- **`canonical.py`** – Defines the universal data structures (`CanonicalInteraction`, `CanonicalMessage`) that all adapters output. Every conversation, regardless of its source framework, ends up in this shape before being formatted.
- **`formatter.py`** – Takes canonical interactions and serialises them into the target fine‑tuning format. Supports OpenAI chat, Alpaca, ShareGPT, prompt‑completion, and preference formats.
- **`storage.py`** – Handles writing formatted data to local files or cloud object stores. Provides a uniform `Path`‑like interface for all backends.
- **`cli.py`** – Implements the `agentscribe` terminal command, allowing users to convert log files or external platform exports into datasets without writing any Python.
- **Adapters (`adapters/*.py`)** – Framework‑specific code that extracts conversation data from each agent runtime and populates a `CanonicalInteraction`. Each adapter knows the exact hook, middleware, or log parser to use for its framework.

---

## Installation

AgentScribe requires Python 3.10 or later. Install the base package with `pip`:

```bash
pip install agentscribe
```

Optional cloud storage dependencies can be installed separately, or all at once:

```bash
pip install agentscribe[s3]       # AWS S3 support
pip install agentscribe[gcs]      # Google Cloud Storage support
pip install agentscribe[azure]    # Azure Blob Storage support
pip install agentscribe[all]      # all storage backends
```

---

## CLI Commands

After installation, the `agentscribe` command is available in your terminal.

### `agentscribe convert`

Convert external logs or exported platform traces into a fine‑tuning dataset.

```bash
agentscribe convert <source> <input> --format <format> --output <path>
```

- `<source>` – The source type. One of: `crewai`, `langgraph`, `agno`, `autogen`, `atomic`, `agentops`, `mlflow`, or `auto` for auto‑detection.
- `<input>` – Path to the log file, directory, or API key file (for AgentOps).
- `--format` – Target dataset format. Options: `openai_chat`, `alpaca`, `sharegpt`, `prompt_completion`, `preference`. Default: `openai_chat`.
- `--output` – Destination path (local or cloud URI).

**Examples:**

```bash
agentscribe convert agentops ./agentops_export.json --format sharegpt --output s3://my-bucket/training/
agentscribe convert crewai ./crew_log.txt --format openai_chat --output ./dataset.jsonl
```

The tool also supports reading from stdin and writing to stdout for pipeline use.

---

## Requirements

**Python:** >= 3.10

**Core dependencies** (installed automatically):

- `click` (CLI)
- `loguru` (Atomic Agents sink support)
- `pyyaml` (configuration)
- `pathlib` (included in Python)
- `json` (included in Python)

**Optional dependencies:**

- `boto3` and `s3fs` for S3 storage (included with `[s3]`)
- `gcsfs` for Google Cloud Storage (included with `[gcs]`)
- `adlfs` for Azure Blob Storage (included with `[azure]`)
- `mlflow` if using the Agno adapter with MLflow autolog
- `requests` and `httpx` for the AgentOps REST API adapter

AgentScribe does not depend on any particular agent framework. The adapters are loaded lazily, so you only need to install the framework you actually use.

---

## License

AgentScribe is released under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contributing

Contributions are welcome! As the project is in its early stages, the best way to get involved is to open an issue to discuss new adapters, formats, or storage backends before submitting a pull request.
