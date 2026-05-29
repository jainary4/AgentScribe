# AgentScribe CLI Reference

AgentScribe ships with a command‑line interface (CLI) that lets you inspect
storage targets, list supported formats, and—most importantly—convert agent
logs and exported datasets into fine‑tuning‑ready JSONL without writing a
single line of Python.

---

## Quick reference

| Command | Purpose |
|---|---|
| `agentscribe` | Base entry point; shows help |
| `agentscribe info` | List supported formats, sources, and storage schemes |
| `agentscribe storage check <URI>` | Resolve a storage target and show backend details |
| `agentscribe convert <source> <input> --output <path> [options]` | Convert logs / records to a training dataset |

The `convert` command is the main workhorse. It accepts raw agent output or
pre‑existing dataset files and writes clean, formatted JSONL.

---

## Global options

Every subcommand respects `-h` / `--help` and the top‑level group also
supports `--version`:

```bash
agentscribe --help
agentscribe --version
```

---

## `info` – show supported capabilities

```bash
agentscribe info
```

Prints:

- The AgentScribe ASCII banner and version.
- The list of output formats: `openai_chat`, `alpaca`, `sharegpt`, `prompt_completion`, `preference`.
- The list of local input sources (`auto`, `json`, `jsonl`, `canonical`, and all five output formats used as inputs).
- The list of adapter sources (`crewai`, `langgraph`, `agno`, `autogen`, `ag2`, `atomic`, `atomic_agents`, `agentops`, `mlflow`, `opentelemetry`, `openinference`, `mcp`).
- The list of supported storage schemes (`file`, `s3`, `r2`, `gs`, `gcs`, `az`, `abfs`, `abfss`, `postgres`, `postgresql`, `pg`).

---

## `storage check` – inspect a storage target

```bash
agentscribe storage check <URI>
```

Resolves the given URI, selects the appropriate storage backend, and reports details.

**Example – local file:**

```bash
agentscribe storage check ./data.jsonl
```

Output:

```
URI: ./data.jsonl
Scheme: file
Backend: local
Path: /home/user/project/data.jsonl
Exists: yes
```

**Example – S3 object:**

```bash
agentscribe storage check s3://my-bucket/training/dataset.jsonl
```

*(requires valid AWS credentials in the environment)*

Output:

```
URI: s3://my-bucket/training/dataset.jsonl
Scheme: s3
Backend: s3
Path: training/dataset.jsonl
Bucket/container: my-bucket
Key: training/dataset.jsonl
Exists: no
```

---

## `convert` – generate a training dataset

### Basic syntax

```bash
agentscribe convert <source> <input_path> --output <output_path> [options]
```

### Arguments

- `<source>` – The format or framework the input data comes from. Can be a local format (`json`, `jsonl`, `canonical`, `openai_chat`, `sharegpt`, `alpaca`, `prompt_completion`, `preference`, or `auto`), or an adapter source (`crewai`, `langgraph`, `agno`, `autogen`, `ag2`, `atomic`, `agentops`, `mlflow`, `opentelemetry`, `openinference`, `mcp`). With `auto`, AgentScribe tries to guess the format from the file extension.

- `<input_path>` – Where to read the input data. A local path like `./logs.txt` or `data.jsonl`, a cloud URI (`s3://…`, `gs://…`, `az://…`), or `-` to read from stdin.

### Required option

- `--output <output_path>` (or `-o`) – Destination for the formatted dataset. A local path or cloud URI, or `-` to write to stdout.

### Options

| Option | Description |
|---|---|
| `--format <fmt>` | Output dataset format. One of `openai_chat` (default), `alpaca`, `sharegpt`, `prompt_completion`, `preference`. |
| `--append` | Append to an existing output file instead of overwriting. |
| `--dataset <name>` | Label for the dataset (stored as metadata in certain backends). |
| `--metadata KEY=VALUE` | Add metadata to the output. Repeatable: `--metadata env=prod --metadata author=jane`. |
| `--skip-invalid` | Skip records that cannot be parsed or normalised, rather than aborting. |

---

## Usage examples

### 1. Convert a JSONL file to OpenAI chat format

```bash
agentscribe convert jsonl ./raw_conversations.jsonl --output ./train.jsonl
```

Reads `raw_conversations.jsonl` (each line a JSON object with a `messages` list) and writes the same structure to `train.jsonl` (default format is `openai_chat`).

---

### 2. Convert an Alpaca dataset to ShareGPT format

```bash
agentscribe convert alpaca ./alpaca_data.json --format sharegpt --output ./sharegpt_data.jsonl
```

The `alpaca` source tells AgentScribe to parse `instruction`, `input`, `output`, and optional `history` fields. The output is a `conversations` list with `from`/`value` pairs.

---

### 3. Convert a CrewAI log file (adapter source)

```bash
agentscribe convert crewai ./crew_run.json --format openai_chat --output ./crew_training.jsonl
```

When an adapter source is used, AgentScribe dispatches the file to the CrewAI adapter's normaliser (e.g., `from_llm_call_context` or the hook context parser). The exact format expected is the structured JSON that the CrewAI adapter can understand (e.g., a log saved by the adapter itself, or a hook context export).

---

### 4. Convert an Agno session export

```bash
agentscribe convert agno ./agno_session.json --format sharegpt --output ./agno_data.jsonl
```

Uses the Agno adapter's `from_session` or `from_run_output` normaliser, depending on the file structure.

---

### 5. Pipe data and write to stdout

```bash
cat logs.jsonl | agentscribe convert auto - --format alpaca --output -
```

Reads JSONL from stdin, auto‑detects the format, converts to Alpaca, and prints the result to stdout. Useful for shell pipelines.

---

### 6. Append to an existing dataset

```bash
agentscribe convert jsonl ./new_batch.jsonl --output ./dataset.jsonl --append
```

The new records are appended to `dataset.jsonl` without touching existing lines.

---

### 7. Skip problematic records

```bash
agentscribe convert auto ./messy_logs.json --output ./clean.jsonl --skip-invalid
```

If a record cannot be parsed, the CLI prints a warning and continues, rather than aborting. Good for cleaning large, imperfect exports.

---

### 8. Add custom metadata

```bash
agentscribe convert crewai ./run.json --output s3://bucket/training/ \
    --format sharegpt \
    --dataset "v2-experiment" \
    --metadata environment=production \
    --metadata pipeline=weekly
```

The dataset label and key‑value pairs are forwarded to the storage backend and may be embedded in the output file (depending on the backend).

---

### 9. Cloud‑native conversion (S3, GCS, Azure)

```bash
# S3 input and output
agentscribe convert jsonl s3://my-logs/raw.jsonl --output s3://my-data/train.jsonl

# Google Cloud Storage
agentscribe convert alpaca gs://bucket/input.json --output gs://bucket/output.jsonl

# Azure Blob
agentscribe convert auto az://container/logs.json --output az://container/dataset.jsonl
```

Authentication uses the standard environment variables or configuration files for each cloud provider (`AWS_ACCESS_KEY_ID`, `GOOGLE_APPLICATION_CREDENTIALS`, `AZURE_STORAGE_CONNECTION_STRING`, etc.). AgentScribe does not manage credentials; it relies on the provider SDKs.

---

## Understanding sources: local vs. adapter

- **Local sources** (`json`, `jsonl`, `canonical`, `openai_chat`, `sharegpt`, `alpaca`, `prompt_completion`, `preference`) are handled directly by the CLI's built‑in parsers. The file is read as JSON or JSONL, and the conversion pipeline runs entirely within the CLI process.

- **Adapter sources** (`crewai`, `agno`, `langgraph`, …) are handled by the corresponding framework adapter. The CLI imports the adapter module (lazily, so the framework is only required when you use that source) and passes the input records to the adapter's normaliser (e.g., `from_run_output`, `from_llm_call_context`). This means you can post‑process logs that were exported from the adapter, or even raw hook context dumps, without running a live agent.

If the necessary framework or dependency is not installed, the CLI will raise a clear error message telling you what to install.

---

## Storage backends

AgentScribe writes output to any of the following, selected automatically from the output path:

| URI prefix | Backend |
|---|---|
| `./`, `/`, or no scheme | Local filesystem |
| `s3://` | Amazon S3 |
| `gs://` or `gcs://` | Google Cloud Storage |
| `az://`, `abfs://`, `abfss://` | Azure Blob Storage |
| `r2://` | Cloudflare R2 |
| `postgres://`, `postgresql://`, `pg://` | PostgreSQL (future support) |

The `storage check` command lets you verify connectivity and permissions before a large conversion job.

---

## Working with compressed files

Input files that end with `.gz` are automatically decompressed. For example:

```bash
agentscribe convert jsonl ./data.jsonl.gz --output ./out.jsonl
```

---

## Notes

- The `convert` command always outputs JSONL (one JSON object per line), regardless of the input format.
- When `--output -` is used, formatted records are written to stdout; log messages and the final summary are printed to stderr, so you can safely pipe the output.
- For adapter sources, the exact input structure depends on the adapter. Refer to the adapter's documentation for the expected log format.
- The `info` command shows which adapter sources are available in your installed version.
