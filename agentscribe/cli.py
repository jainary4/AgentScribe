"""Command line interface for AgentScribe."""

from __future__ import annotations

import gzip
import json
import sys
from collections.abc import Iterable, Iterator, Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import click

from agentscribe.adapters.utils.registry import adapter_record_to_interactions as dispatch_adapter_record
from agentscribe.core.canonical import CanonicalInteraction
from agentscribe.core.formatter import FormatValidationError, available_formats, format_messages
from agentscribe.storage import StorageError, StorageURI, get_backend, read_jsonl, write_jsonl


FORMATS = tuple(available_formats())
LOCAL_SOURCES = ("auto", "json", "jsonl", "canonical", "openai_chat", "sharegpt", "alpaca", "prompt_completion", "preference")
ADAPTER_SOURCES = (
	"crewai",
	"langgraph",
	"agno",
	"autogen",
	"ag2",
	"atomic",
	"atomic_agents",
	"agentops",
	"mlflow",
	"opentelemetry",
	"openinference",
	"mcp",
)
STORAGE_SCHEMES = ("file", "s3", "r2", "gs", "gcs", "az", "abfs", "abfss", "postgres", "postgresql", "pg")

AGENTSCRIBE_BANNER = r"""
    _    ____ _____ _   _ _____ ____   ____ ____  ___ ____  _____
   / \  / ___| ____| \ | |_   _/ ___| / ___|  _ \|_ _| __ )| ____|
  / _ \| |  _|  _| |  \| | | | \___ \| |   | |_) || ||  _ \|  _|
 / ___ \ |_| | |___| |\  | | |  ___) | |___|  _ < | || |_) | |___
/_/   \_\____|_____|_| \_| |_| |____/ \____|_| \_\___|____/|_____|
""".strip("\n")

SHAREGPT_TO_CANONICAL = {
	"human": "user",
	"gpt": "assistant",
	"function_call": "tool_call",
	"observation": "tool_response",
}


def _package_version() -> str:
	try:
		return version("agentscribe")
	except PackageNotFoundError:
		return "0.1.0"


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(_package_version(), prog_name="agentscribe")
def main() -> None:
	"""Capture, inspect, and convert AgentScribe fine-tuning datasets."""


@main.command()
def info() -> None:
	"""Show supported formats, sources, and storage schemes."""

	click.echo(AGENTSCRIBE_BANNER)
	click.echo(f"AgentScribe {_package_version()}")
	click.echo(f"Formats: {', '.join(FORMATS)}")
	click.echo(f"Ready sources: {', '.join(LOCAL_SOURCES)}")
	click.echo(f"Adapter sources: {', '.join(ADAPTER_SOURCES)}")
	click.echo(f"Storage schemes: {', '.join(STORAGE_SCHEMES)}")
	click.echo("Default storage: local JSONL files when no URI scheme is provided")


@main.group(name="storage")
def storage_group() -> None:
	"""Inspect storage targets and backend resolution."""


@storage_group.command(name="check")
@click.argument("target")
def storage_check(target: str) -> None:
	"""Resolve a storage target and report the selected backend."""

	try:
		parsed_uri = StorageURI.parse(target)
		backend = get_backend(target)
	except StorageError as exc:
		raise click.ClickException(str(exc)) from exc

	click.echo(f"URI: {target}")
	click.echo(f"Scheme: {parsed_uri.scheme}")
	click.echo(f"Backend: {backend.name}")
	click.echo(f"Path: {parsed_uri.path}")
	if parsed_uri.bucket:
		click.echo(f"Bucket/container: {parsed_uri.bucket}")
	if parsed_uri.key:
		click.echo(f"Key: {parsed_uri.key}")

	try:
		exists = backend.exists(target)
	except Exception as exc:
		click.echo(f"Exists: unknown ({exc})")
	else:
		click.echo(f"Exists: {'yes' if exists else 'no'}")


@main.command()
@click.argument("source")
@click.argument("input_path")
@click.option(
	"--format",
	"format_name",
	type=click.Choice(FORMATS),
	default="openai_chat",
	show_default=True,
	help="Output dataset format.",
)
@click.option("--output", "output_path", required=True, help="Output JSONL path, storage URI, or '-' for stdout.")
@click.option("--append", is_flag=True, help="Append to an existing JSONL output instead of replacing it.")
@click.option("--dataset", help="Dataset label for storage backends that preserve metadata.")
@click.option(
	"--metadata",
	multiple=True,
	metavar="KEY=VALUE",
	help="Metadata pair for storage backends that preserve metadata. May be repeated.",
)
@click.option("--skip-invalid", is_flag=True, help="Skip records that cannot be normalized.")
def convert(
	source: str,
	input_path: str,
	format_name: str,
	output_path: str,
	append: bool,
	dataset: str | None,
	metadata: tuple[str, ...],
	skip_invalid: bool,
) -> None:
	"""Convert JSON/JSONL/canonical records into a training dataset."""

	normalized_source = source.lower().replace("-", "_")
	_ensure_supported_source(normalized_source)
	parsed_metadata = _parse_metadata(metadata)

	formatted_records = _format_records(
		_load_records(normalized_source, input_path),
		source=normalized_source,
		format_name=format_name,
		skip_invalid=skip_invalid,
	)

	if output_path == "-":
		count = _write_stdout(formatted_records)
		click.echo(f"Wrote {count} records to stdout", err=True)
		return

	try:
		result = write_jsonl(
			output_path,
			formatted_records,
			mode="a" if append else "w",
			dataset=dataset,
			format_name=format_name,
			metadata=parsed_metadata,
		)
	except StorageError as exc:
		raise click.ClickException(str(exc)) from exc

	click.echo(f"Wrote {result.records_written} records to {result.uri} using {result.backend} storage")


def _ensure_supported_source(source: str) -> None:

	"""Check that the source identifier is a known local format or a known adapter (even if not yet implemented). 
	Raise a helpful error otherwise.
	Usage: Called at the start of convert after normalizing the source string."""

	if source in LOCAL_SOURCES:
		return
	if source in ADAPTER_SOURCES:
		return
	accepted = ", ".join((*LOCAL_SOURCES, *ADAPTER_SOURCES))
	raise click.ClickException(f"Unknown source `{source}`. Expected one of: {accepted}")


def _load_records(source: str, input_path: str) -> Iterator[Mapping[str, Any]]:

	"""Read and parse input data (file or stdin) into an iterator of record dictionaries, 
	using the source hint to select a parser.
	Usage: Called by convert to turn the raw input into record dictionaries."""

	if input_path == "-":
		yield from _parse_json_documents(sys.stdin.read())
		return

	if source == "jsonl" or (source == "auto" and _looks_like_jsonl(input_path)):
		yield from _ensure_mapping_records(read_jsonl(input_path))
		return

	payload = _read_json_document(input_path)
	if isinstance(payload, list):
		yield from _ensure_mapping_records(payload)
	else:
		yield from _ensure_mapping_records([payload])


def _parse_json_documents(text: str) -> Iterator[Mapping[str, Any]]:

	"""Parse a raw text string into an iterator of JSON objects, 
	trying first as a full JSON document, then as JSONL line by line
	Usage: only when input_path == '-' """

	stripped = text.strip()
	if not stripped:
		return

	try:
		payload = json.loads(stripped)
	except json.JSONDecodeError:
		for line_number, line in enumerate(stripped.splitlines(), start=1):
			if not line.strip():
				continue
			try:
				record = json.loads(line)
			except json.JSONDecodeError as exc:
				raise click.ClickException(f"Invalid JSONL at stdin line {line_number}: {exc}") from exc
			yield from _ensure_mapping_records([record])
		return

	if isinstance(payload, list):
		yield from _ensure_mapping_records(payload)
	else:
		yield from _ensure_mapping_records([payload])


def _read_json_document(input_path: str) -> Any:

	"""Open and read a JSON file from any supported storage backend, 
	transparently handling gzip compression

	Usage: Called by _load_records when input is not JSONL and not stdin"""

	try:
		backend = get_backend(input_path)
	except StorageError as exc:
		raise click.ClickException(str(exc)) from exc

	if not backend.supports_file_objects:
		raise click.ClickException(f"{backend.name} storage cannot be used as a JSON document input")

	try:
		with backend.open(input_path, "rb") as raw_stream:
			if _looks_gzipped(input_path):
				with gzip.GzipFile(fileobj=raw_stream, mode="rb") as stream:
					return json.loads(stream.read().decode("utf-8"))
			return json.loads(raw_stream.read().decode("utf-8"))
	except json.JSONDecodeError as exc:
		raise click.ClickException(f"Invalid JSON document in {input_path}: {exc}") from exc


def _ensure_mapping_records(records: Iterable[Any]) -> Iterator[Mapping[str, Any]]:

	"""Validate that every item in an iterable is a dictionary, raising an error for the first non‑mapping item found"""

	for index, record in enumerate(records, start=1):
		if not isinstance(record, Mapping):
			raise click.ClickException(f"Record {index} is not a JSON object")
		yield record


def _format_records(
	records: Iterable[Mapping[str, Any]],
	*,
	source: str,
	format_name: str,
	skip_invalid: bool,
) -> Iterator[Mapping[str, Any]]:
	
	"""Convert each raw record to a canonical message list, 
	then format to the desired output, optionally skipping invalid records

	Usage: Core loop inside the convert command"""
	for index, record in enumerate(records, start=1):
		try:
			if source in ADAPTER_SOURCES:
				for messages in _adapter_record_to_message_batches(record, source=source):
					yield _format_messages(messages, format_name)
			else:
				messages = _record_to_messages(record, source=source)
				yield _format_messages(messages, format_name)
		except click.ClickException as exc:
			if not skip_invalid:
				raise click.ClickException(f"Record {index}: {exc.message}") from exc
			click.echo(f"Skipping record {index}: {exc.message}", err=True)


def _adapter_record_to_message_batches(record: Mapping[str, Any], *, source: str) -> list[list[dict[str, Any]]]:

	"""Convert an adapter export record into one or more canonical message lists.
	Usage: Called by _format_records for framework/platform sources."""

	if "messages" in record and "source_framework" in record:
		return [CanonicalInteraction.from_dict(dict(record)).to_dict()["messages"]]

	interactions = _adapter_record_to_interactions(record, source=source)
	return [interaction.to_dict()["messages"] for interaction in interactions]


def _adapter_record_to_interactions(record: Mapping[str, Any], *, source: str) -> list[CanonicalInteraction]:

	"""Dispatch JSON exports to the relevant optional-dependency adapter."""

	return dispatch_adapter_record(record, source=source)


def _record_to_messages(record: Mapping[str, Any], *, source: str) -> list[dict[str, Any]]:

	"""Transform a raw record into a canonical list of message dicts, auto‑detecting the input structure
	Usage:Called by _format_records for every input record."""

	if "messages" in record:
		return [_normalize_message(message) for message in _expect_list(record["messages"], "messages")]

	if "conversations" in record:
		return [_normalize_message(message) for message in _expect_list(record["conversations"], "conversations")]

	if source == "alpaca" or _looks_like_alpaca(record):
		return _alpaca_to_messages(record)

	if source in {"prompt_completion", "preference"} or "prompt" in record:
		return _prompt_record_to_messages(record)

	raise click.ClickException("could not find messages, conversations, Alpaca fields, or prompt/completion fields")


def _normalize_message(message: Any) -> dict[str, Any]:

	"""Convert a raw message dict into a canonical dict with role and content, preserving any extra fields
	Usage: Used by _record_to_messages whenever we encounter a message in a list."""

	if not isinstance(message, Mapping):
		raise click.ClickException("message entries must be JSON objects") # checks if the message is a valid json object( dictionary)


	role = message.get("role") or message.get("from")
	if not role:
		raise click.ClickException("message is missing role/from")

	role = SHAREGPT_TO_CANONICAL.get(str(role), str(role))
	content = message.get("content", message.get("value", "")) # maps the role to the canconical format
	normalized = {"role": role, "content": _coerce_text(content)}

	for key, value in message.items():
		if key not in {"role", "from", "content", "value"}:
			normalized[key] = value
	return normalized


def _alpaca_to_messages(record: Mapping[str, Any]) -> list[dict[str, Any]]:

	"""Convert an Alpaca‑style record (instruction, output, optional history) into a flat canonical message list
	Usage: Used by _record_to_messages when Alpaca format is detected"""

	messages: list[dict[str, Any]] = [] #Start with an empty list

	for turn in record.get("history") or []: #If the record has a history field, each turn must be a pair of strings (user, assistant)
		if not isinstance(turn, (list, tuple)) or len(turn) != 2:
			raise click.ClickException("Alpaca history entries must be [user, assistant] pairs")
		messages.append({"role": "user", "content": _coerce_text(turn[0])})
		messages.append({"role": "assistant", "content": _coerce_text(turn[1])})

	instruction = _coerce_text(record.get("instruction", ""))
	input_text = _coerce_text(record.get("input", ""))
	user_content = f"{instruction}\n\n{input_text}".strip() if input_text else instruction
	if user_content:
		messages.append({"role": "user", "content": user_content}) #Combine the instruction and optional input into one user message

	output = _coerce_text(record.get("output", ""))
	if output:
		messages.append({"role": "assistant", "content": output})
	return messages


def _prompt_record_to_messages(record: Mapping[str, Any]) -> list[dict[str, Any]]:

	"""Convert a prompt‑completion or preference record into a canonical message list.
	Usage: Called by _record_to_messages for prompt‑based records."""
	prompt = record.get("prompt", "")
	if isinstance(prompt, list):
		messages = [_normalize_message(message) for message in prompt]
	else:
		messages = [{"role": "user", "content": _coerce_text(prompt)}]

	#If the prompt field is already a list of messages (like in DPO or OpenAI format), we normalize each one. 
	# Otherwise, treat the prompt as a single user string

	if "completion" in record:
		messages.append({"role": "assistant", "content": _coerce_text(record.get("completion", ""))})
		return messages

	chosen = record.get("chosen", "")
	if isinstance(chosen, list):
		messages.extend(_normalize_message(message) for message in chosen)
	elif chosen:
		messages.append({"role": "assistant", "content": _coerce_text(chosen)})
	return messages


def _format_messages(messages: list[dict[str, Any]], format_name: str) -> Mapping[str, Any]:

	"""Route a canonical message list to the shared formatter for the chosen format.

	Usage: Called by _format_records after normalizing a record. Delegates to
	agentscribe.core.formatter so the CLI and the adapters share one spec-compliant
	implementation."""

	try:
		return format_messages(messages, format_name)
	except FormatValidationError as exc:
		raise click.ClickException(str(exc)) from exc
	except ValueError as exc:
		raise click.ClickException(
			f"Unsupported format `{format_name}`. Supported: {', '.join(available_formats())}"
		) from exc


def _write_stdout(records: Iterable[Mapping[str, Any]]) -> int:
	""" Write formatted records as compact JSONL to stdout and return the count
	Usage: Uses click.echo to print each record as a single line. s
	eparators=(",", ":") removes spaces, making the output more compact. Returns the number of records written."""

	count = 0
	for record in records:
		click.echo(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
		count += 1
	return count


def _parse_metadata(entries: tuple[str, ...]) -> Mapping[str, str]:

	"""Parse repeated KEY=VALUE strings into a metadata dictionary
	Usage : Called at the beginning of convert to process the --metadata options"""

	metadata: dict[str, str] = {}
	for entry in entries:
		if "=" not in entry:
			raise click.BadParameter("metadata must be provided as KEY=VALUE", param_hint="--metadata")
		key, value = entry.split("=", 1)
		if not key:
			raise click.BadParameter("metadata keys cannot be empty", param_hint="--metadata")
		metadata[key] = value
	return metadata


def _expect_list(value: Any, field_name: str) -> list[Any]:

	""" Assert that a value is a list; raise a readable error if not
	Usage: Used in _record_to_messages to validate messages and conversations fields"""

	if not isinstance(value, list):
		raise click.ClickException(f"{field_name} must be a list")
	return value


def _looks_like_alpaca(record: Mapping[str, Any]) -> bool:

	""" Return True if the record contains both instruction and output keys"""

	return "instruction" in record and "output" in record


def _looks_like_jsonl(input_path: str) -> bool:

	""" Check if a file path likely points to a JSONL file by its extension"""

	suffixes = Path(input_path).suffixes
	return ".jsonl" in suffixes or ".ndjson" in suffixes


def _looks_gzipped(input_path: str) -> bool:

	"""Return True if the file path ends with .gz"""

	return ".gz" in Path(input_path).suffixes


def _coerce_text(value: Any) -> str:

	"""Safely convert any value to a string: None becomes empty, 
	strings pass through, anything else is serialised to JSON"""

	if value is None:
		return ""
	if isinstance(value, str):
		return value
	return json.dumps(value, ensure_ascii=False, default=str)


if __name__ == "__main__":
	main()
