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

from agentscribe.storage import StorageError, StorageURI, get_backend, read_jsonl, write_jsonl


FORMATS = ("openai_chat", "alpaca", "sharegpt", "prompt_completion", "preference")
LOCAL_SOURCES = ("auto", "json", "jsonl", "canonical", "openai_chat", "sharegpt", "alpaca", "prompt_completion", "preference")
ADAPTER_SOURCES = ("crewai", "langgraph", "agno", "autogen", "atomic", "agentops", "mlflow")
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
CANONICAL_TO_SHAREGPT = {
	"system": "system",
	"user": "human",
	"assistant": "gpt",
	"tool_call": "function_call",
	"tool_response": "observation",
	"tool": "observation",
}
OPENAI_ROLE_MAP = {
	"tool_response": "tool",
	"tool_call": "assistant",
}
OPENAI_MESSAGE_EXTRAS = {"name", "tool_call_id", "tool_calls", "function_call", "weight"}


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
	if source in LOCAL_SOURCES:
		return
	if source in ADAPTER_SOURCES:
		raise click.ClickException(
			f"The `{source}` adapter is not implemented in the CLI yet. "
			"Use `json`, `jsonl`, or `canonical` input for now."
		)
	accepted = ", ".join((*LOCAL_SOURCES, *ADAPTER_SOURCES))
	raise click.ClickException(f"Unknown source `{source}`. Expected one of: {accepted}")


def _load_records(source: str, input_path: str) -> Iterator[Mapping[str, Any]]:
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
	for index, record in enumerate(records, start=1):
		try:
			messages = _record_to_messages(record, source=source)
			yield _format_messages(messages, format_name)
		except click.ClickException as exc:
			if not skip_invalid:
				raise click.ClickException(f"Record {index}: {exc.message}") from exc
			click.echo(f"Skipping record {index}: {exc.message}", err=True)


def _record_to_messages(record: Mapping[str, Any], *, source: str) -> list[dict[str, Any]]:
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
	if not isinstance(message, Mapping):
		raise click.ClickException("message entries must be JSON objects")

	role = message.get("role") or message.get("from")
	if not role:
		raise click.ClickException("message is missing role/from")

	role = SHAREGPT_TO_CANONICAL.get(str(role), str(role))
	content = message.get("content", message.get("value", ""))
	normalized = {"role": role, "content": _coerce_text(content)}

	for key, value in message.items():
		if key not in {"role", "from", "content", "value"}:
			normalized[key] = value
	return normalized


def _alpaca_to_messages(record: Mapping[str, Any]) -> list[dict[str, Any]]:
	messages: list[dict[str, Any]] = []

	for turn in record.get("history") or []:
		if not isinstance(turn, (list, tuple)) or len(turn) != 2:
			raise click.ClickException("Alpaca history entries must be [user, assistant] pairs")
		messages.append({"role": "user", "content": _coerce_text(turn[0])})
		messages.append({"role": "assistant", "content": _coerce_text(turn[1])})

	instruction = _coerce_text(record.get("instruction", ""))
	input_text = _coerce_text(record.get("input", ""))
	user_content = f"{instruction}\n\n{input_text}".strip() if input_text else instruction
	if user_content:
		messages.append({"role": "user", "content": user_content})

	output = _coerce_text(record.get("output", ""))
	if output:
		messages.append({"role": "assistant", "content": output})
	return messages


def _prompt_record_to_messages(record: Mapping[str, Any]) -> list[dict[str, Any]]:
	prompt = record.get("prompt", "")
	if isinstance(prompt, list):
		messages = [_normalize_message(message) for message in prompt]
	else:
		messages = [{"role": "user", "content": _coerce_text(prompt)}]

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
	if format_name == "openai_chat":
		return {"messages": [_to_openai_message(message) for message in messages]}
	if format_name == "sharegpt":
		return _to_sharegpt(messages)
	if format_name == "alpaca":
		return _to_alpaca(messages)
	if format_name == "prompt_completion":
		return _to_prompt_completion(messages)
	if format_name == "preference":
		return _to_preference(messages)
	raise click.ClickException(f"Unsupported format `{format_name}`")


def _to_openai_message(message: Mapping[str, Any]) -> dict[str, Any]:
	role = OPENAI_ROLE_MAP.get(str(message["role"]), str(message["role"]))
	result = {"role": role, "content": message.get("content", "")}
	for key in OPENAI_MESSAGE_EXTRAS:
		if key in message:
			result[key] = message[key]
	return result


def _to_sharegpt(messages: list[dict[str, Any]]) -> Mapping[str, Any]:
	conversations = []
	system = ""
	for message in messages:
		role = str(message["role"])
		if role == "system" and not system:
			system = _coerce_text(message.get("content", ""))
		conversations.append(
			{
				"from": CANONICAL_TO_SHAREGPT.get(role, role),
				"value": _coerce_text(message.get("content", "")),
			}
		)
	return {"conversations": conversations, "system": system}


def _to_alpaca(messages: list[dict[str, Any]]) -> Mapping[str, Any]:
	user_messages = [message for message in messages if message["role"] == "user"]
	assistant_messages = [message for message in messages if message["role"] == "assistant"]
	system_message = next((message for message in messages if message["role"] == "system"), None)

	instruction = _coerce_text(user_messages[-1].get("content", "")) if user_messages else ""
	if system_message:
		instruction = f"{_coerce_text(system_message.get('content', ''))}\n\n{instruction}".strip()

	result: dict[str, Any] = {
		"instruction": instruction,
		"input": "",
		"output": _coerce_text(assistant_messages[-1].get("content", "")) if assistant_messages else "",
	}

	history_pairs = list(zip(user_messages[:-1], assistant_messages[:-1]))
	if history_pairs:
		result["history"] = [
			[_coerce_text(user_message.get("content", "")), _coerce_text(assistant_message.get("content", ""))]
			for user_message, assistant_message in history_pairs
		]
	return result


def _to_prompt_completion(messages: list[dict[str, Any]]) -> Mapping[str, Any]:
	user_message = next((message for message in messages if message["role"] == "user"), None)
	assistant_message = next((message for message in reversed(messages) if message["role"] == "assistant"), None)
	return {
		"prompt": _coerce_text(user_message.get("content", "")) if user_message else "",
		"completion": _coerce_text(assistant_message.get("content", "")) if assistant_message else "",
	}


def _to_preference(messages: list[dict[str, Any]]) -> Mapping[str, Any]:
	user_content = " ".join(_coerce_text(message.get("content", "")) for message in messages if message["role"] == "user")
	assistant_content = " ".join(
		_coerce_text(message.get("content", "")) for message in messages if message["role"] == "assistant"
	)
	return {
		"prompt": [{"role": "user", "content": user_content}],
		"chosen": [{"role": "assistant", "content": assistant_content}],
		"rejected": [{"role": "assistant", "content": ""}],
	}


def _write_stdout(records: Iterable[Mapping[str, Any]]) -> int:
	count = 0
	for record in records:
		click.echo(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
		count += 1
	return count


def _parse_metadata(entries: tuple[str, ...]) -> Mapping[str, str]:
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
	if not isinstance(value, list):
		raise click.ClickException(f"{field_name} must be a list")
	return value


def _looks_like_alpaca(record: Mapping[str, Any]) -> bool:
	return "instruction" in record and "output" in record


def _looks_like_jsonl(input_path: str) -> bool:
	suffixes = Path(input_path).suffixes
	return ".jsonl" in suffixes or ".ndjson" in suffixes


def _looks_gzipped(input_path: str) -> bool:
	return ".gz" in Path(input_path).suffixes


def _coerce_text(value: Any) -> str:
	if value is None:
		return ""
	if isinstance(value, str):
		return value
	return json.dumps(value, ensure_ascii=False, default=str)


if __name__ == "__main__":
	main()
