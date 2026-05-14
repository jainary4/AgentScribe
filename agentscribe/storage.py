"""Storage primitives for AgentScribe datasets.

The storage layer is intentionally small and dependency-light. Local writes work
with the Python standard library, object stores use ``fsspec`` when the matching
extra is installed, and Postgres uses a lazy ``psycopg`` import only when a
Postgres URI is selected.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, BinaryIO, Literal
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse
from uuid import UUID


Compression = Literal["infer", "none", "gzip"]
WriteMode = Literal["w", "a"]
JsonRecord = Mapping[str, Any] | Sequence[Any]


class StorageError(RuntimeError):
	"""Base exception for storage failures."""


class UnsupportedBackendError(StorageError):
	"""Raised when no backend is registered for a URI scheme."""


class MissingDependencyError(StorageError):
	"""Raised when an optional storage dependency is not installed."""


@dataclass(frozen=True)
class StorageURI:
	"""Parsed storage URI with normalized access to common parts."""

	raw: str
	scheme: str
	path: str
	bucket: str | None = None
	key: str | None = None
	query: Mapping[str, tuple[str, ...]] | None = None

	@classmethod
	def parse(cls, uri: str | os.PathLike[str]) -> "StorageURI":
		raw = os.fspath(uri)
		parsed = urlparse(raw)

		if not parsed.scheme or len(parsed.scheme) == 1:
			return cls(raw=raw, scheme="file", path=raw, query={})

		query = {key: tuple(values) for key, values in parse_qs(parsed.query).items()}

		if parsed.scheme == "file":
			return cls(raw=raw, scheme="file", path=parsed.path, query=query)

		if parsed.scheme in {"s3", "r2", "gs", "gcs", "az", "abfs", "abfss"}:
			key = parsed.path.lstrip("/")
			return cls(
				raw=raw,
				scheme=parsed.scheme,
				path=f"{parsed.netloc}/{key}" if key else parsed.netloc,
				bucket=parsed.netloc,
				key=key,
				query=query,
			)

		return cls(raw=raw, scheme=parsed.scheme, path=parsed.path, query=query)

	@property
	def suffixes(self) -> tuple[str, ...]:
		return tuple(Path(self.key or self.path).suffixes)


@dataclass(frozen=True)
class WriteResult:
	"""Result returned by storage write operations."""

	uri: str
	records_written: int
	bytes_written: int | None = None
	backend: str = "unknown"


class JsonLineSerializer:
	"""Fast JSONL serializer with an ``orjson`` fast path when available."""

	def __init__(self, *, ensure_ascii: bool = False) -> None:
		self.ensure_ascii = ensure_ascii
		try:
			import orjson  # type: ignore[import-not-found]
		except ImportError:
			self._orjson = None
		else:
			self._orjson = orjson

	def dumps(self, record: Any) -> bytes:
		normalized = normalize_record(record)
		if self._orjson is not None:
			return self._orjson.dumps(normalized, default=_json_default)

		return json.dumps(
			normalized,
			ensure_ascii=self.ensure_ascii,
			separators=(",", ":"),
			default=_json_default,
		).encode("utf-8")


class StorageBackend(ABC):
	"""Base class for object and database storage backends."""

	name: str = "storage"
	supports_file_objects: bool = True

	@abstractmethod
	def open(self, uri: str, mode: str = "rb", *, atomic: bool = False) -> BinaryIO:
		"""Open a binary file-like object for the URI."""

	def exists(self, uri: str) -> bool:
		raise NotImplementedError(f"{self.name} does not implement exists()")

	def write_records(
		self,
		uri: str,
		records: Iterable[Any],
		*,
		serializer: JsonLineSerializer | None = None,
		compression: Compression = "infer",
		mode: WriteMode = "w",
		batch_size: int = 1000,
		dataset: str | None = None,
		format_name: str | None = None,
		metadata: Mapping[str, Any] | None = None,
	) -> WriteResult:
		del batch_size, dataset, format_name, metadata
		return _write_jsonl_to_file_backend(
			backend=self,
			uri=uri,
			records=records,
			serializer=serializer or JsonLineSerializer(),
			compression=compression,
			mode=mode,
		)


class LocalStorageBackend(StorageBackend):
	"""Storage backend for local paths and ``file://`` URIs."""

	name = "local"


	@contextmanager
	def open(self, uri: str, mode: str = "rb", *, atomic: bool = False) -> Iterator[BinaryIO]:
		parsed_uri = StorageURI.parse(uri)
		target = Path(parsed_uri.path).expanduser()

		if _is_write_mode(mode):
			target.parent.mkdir(parents=True, exist_ok=True)

		if atomic and _is_replace_mode(mode):
			temp_file = tempfile.NamedTemporaryFile(
				mode=mode,
				delete=False,
				dir=str(target.parent),
				prefix=f".{target.name}.",
				suffix=".tmp",
			)
			temp_path = Path(temp_file.name)
			try:
				with temp_file:
					yield temp_file
				os.replace(temp_path, target)
			except Exception:
				temp_path.unlink(missing_ok=True)
				raise
			return

		with target.open(mode) as file_obj:
			yield file_obj

	def exists(self, uri: str) -> bool:
		parsed_uri = StorageURI.parse(uri)
		return Path(parsed_uri.path).expanduser().exists()


class FSSpecStorageBackend(StorageBackend):
	"""Storage backend for S3, R2, GCS, Azure Blob, and similar object stores."""

	name = "fsspec"

	def __init__(self, *, storage_options: Mapping[str, Any] | None = None) -> None:
		self.storage_options = dict(storage_options or {})

	@contextmanager
	def open(self, uri: str, mode: str = "rb", *, atomic: bool = False) -> Iterator[BinaryIO]:
		del atomic
		fsspec = _import_fsspec()
		normalized_uri, options = _normalize_fsspec_target(uri, self.storage_options)

		with fsspec.open(normalized_uri, mode=mode, **options) as file_obj:
			yield file_obj

	def exists(self, uri: str) -> bool:
		fsspec = _import_fsspec()
		normalized_uri, options = _normalize_fsspec_target(uri, self.storage_options)
		filesystem, path = fsspec.core.url_to_fs(normalized_uri, **options)
		return filesystem.exists(path)


class PostgresStorageBackend(StorageBackend):
	"""JSONB row-store backend for Postgres-compatible databases."""

	name = "postgres"
	supports_file_objects = False

	def __init__(self, *, storage_options: Mapping[str, Any] | None = None) -> None:
		self.storage_options = dict(storage_options or {})

	def open(self, uri: str, mode: str = "rb", *, atomic: bool = False) -> BinaryIO:
		del uri, mode, atomic
		raise UnsupportedBackendError("Postgres storage writes records, not file objects")

	def write_records(
		self,
		uri: str,
		records: Iterable[Any],
		*,
		serializer: JsonLineSerializer | None = None,
		compression: Compression = "infer",
		mode: WriteMode = "w",
		batch_size: int = 1000,
		dataset: str | None = None,
		format_name: str | None = None,
		metadata: Mapping[str, Any] | None = None,
	) -> WriteResult:
		del compression, mode
		serializer = serializer or JsonLineSerializer()
		parsed_uri = urlparse(uri)
		query = parse_qs(parsed_uri.query)

		schema_name = _option_value("schema", query, self.storage_options, "public")
		table_name = _option_value("table", query, self.storage_options, "agentscribe_records")
		dataset_name = dataset or _option_value("dataset", query, self.storage_options, None)
		format_value = format_name or _option_value("format", query, self.storage_options, None)
		should_create_table = bool(self.storage_options.get("create_table", True))

		schema_sql = _quote_identifier(schema_name)
		table_sql = _quote_identifier(table_name)
		connection = self._connect(uri)
		close_connection = "connection" not in self.storage_options
		records_written = 0

		try:
			with connection.cursor() as cursor:
				if should_create_table:
					cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {schema_sql}")
					cursor.execute(
						f"""
						CREATE TABLE IF NOT EXISTS {schema_sql}.{table_sql} (
							id bigserial PRIMARY KEY,
							dataset text,
							format text,
							payload jsonb NOT NULL,
							metadata jsonb NOT NULL DEFAULT '{{}}'::jsonb,
							created_at timestamptz NOT NULL DEFAULT now()
						)
						"""
					)

				insert_sql = f"""
					INSERT INTO {schema_sql}.{table_sql} (dataset, format, payload, metadata)
					VALUES (%s, %s, %s::jsonb, %s::jsonb)
				"""

				for batch in _batched(records, batch_size):
					rows = [
						(
							dataset_name,
							format_value,
							serializer.dumps(record).decode("utf-8"),
							json.dumps(metadata or {}, separators=(",", ":")),
						)
						for record in batch
					]
					cursor.executemany(insert_sql, rows)
					records_written += len(rows)

			connection.commit()
		except Exception:
			connection.rollback()
			raise
		finally:
			if close_connection:
				connection.close()

		return WriteResult(uri=_redact_connection_uri(uri), records_written=records_written, backend=self.name)

	def _connect(self, uri: str) -> Any:
		connection = self.storage_options.get("connection")
		if connection is not None:
			return connection

		try:
			import psycopg  # type: ignore[import-not-found]
		except ImportError as exc:
			raise MissingDependencyError(
				"Postgres storage requires psycopg. Install it with `pip install psycopg[binary]`."
			) from exc

		dsn = _strip_storage_query_params(uri)
		return psycopg.connect(dsn, **dict(self.storage_options.get("connect_kwargs", {})))


BackendFactory = Callable[[Mapping[str, Any] | None], StorageBackend]
_BACKEND_REGISTRY: dict[str, BackendFactory] = {}


def register_backend(scheme: str | Sequence[str], factory: BackendFactory) -> None:
	"""Register a custom storage backend factory for one or more URI schemes."""

	schemes = (scheme,) if isinstance(scheme, str) else scheme
	for item in schemes:
		_BACKEND_REGISTRY[item.lower()] = factory


def get_backend(uri: str | os.PathLike[str], *, storage_options: Mapping[str, Any] | None = None) -> StorageBackend:
	"""Resolve the backend that should handle ``uri``."""

	parsed_uri = StorageURI.parse(uri)
	factory = _BACKEND_REGISTRY.get(parsed_uri.scheme)
	if factory is None:
		raise UnsupportedBackendError(f"No storage backend registered for scheme `{parsed_uri.scheme}`")
	return factory(storage_options)


def write_jsonl(
	uri: str | os.PathLike[str],
	records: Iterable[Any],
	*,
	storage_options: Mapping[str, Any] | None = None,
	serializer: JsonLineSerializer | None = None,
	compression: Compression = "infer",
	mode: WriteMode = "w",
	batch_size: int = 1000,
	dataset: str | None = None,
	format_name: str | None = None,
	metadata: Mapping[str, Any] | None = None,
) -> WriteResult:
	"""Write records to a JSONL object or Postgres JSONB row store.

	Records are streamed one at a time for file/object backends and inserted in
	batches for Postgres. Any object with ``to_dict()``, Pydantic
	``model_dump()``, or dataclass fields is converted automatically.
	"""

	backend = get_backend(uri, storage_options=storage_options)
	return backend.write_records(
		os.fspath(uri),
		records,
		serializer=serializer,
		compression=compression,
		mode=mode,
		batch_size=batch_size,
		dataset=dataset,
		format_name=format_name,
		metadata=metadata,
	)


def write_json(
	uri: str | os.PathLike[str],
	payload: Any,
	*,
	storage_options: Mapping[str, Any] | None = None,
	serializer: JsonLineSerializer | None = None,
	compression: Compression = "infer",
) -> WriteResult:
	"""Write one JSON document to a local or object-store URI."""

	serializer = serializer or JsonLineSerializer()
	backend = get_backend(uri, storage_options=storage_options)
	if not backend.supports_file_objects:
		raise UnsupportedBackendError(f"{backend.name} does not support JSON document writes")

	json_bytes = serializer.dumps(payload) + b"\n"
	bytes_written = 0

	with backend.open(os.fspath(uri), "wb", atomic=True) as raw_stream:
		stream = _wrap_compression(raw_stream, StorageURI.parse(uri), compression)
		try:
			stream.write(json_bytes)
			bytes_written += len(json_bytes)
		finally:
			if stream is not raw_stream:
				stream.close()

	return WriteResult(uri=os.fspath(uri), records_written=1, bytes_written=bytes_written, backend=backend.name)


def read_jsonl(
	uri: str | os.PathLike[str],
	*,
	storage_options: Mapping[str, Any] | None = None,
	compression: Compression = "infer",
) -> Iterator[Any]:
	"""Read JSONL records from a local or object-store URI."""

	backend = get_backend(uri, storage_options=storage_options)
	if not backend.supports_file_objects:
		raise UnsupportedBackendError(f"{backend.name} does not support JSONL reads")

	with backend.open(os.fspath(uri), "rb") as raw_stream:
		stream = _wrap_decompression(raw_stream, StorageURI.parse(uri), compression)
		try:
			for line in stream:
				if line.strip():
					yield json.loads(line)
		finally:
			if stream is not raw_stream:
				stream.close()


def normalize_record(record: Any) -> Any:
	"""Normalize common model objects into JSON-serializable structures."""

	if hasattr(record, "model_dump"):
		return record.model_dump(mode="json")
	if hasattr(record, "to_dict"):
		return record.to_dict()
	if is_dataclass(record) and not isinstance(record, type):
		return asdict(record)
	if isinstance(record, Path):
		return str(record)
	if isinstance(record, (date, datetime)):
		return record.isoformat()
	if isinstance(record, (Decimal, UUID)):
		return str(record)
	return record


def _write_jsonl_to_file_backend(
	*,
	backend: StorageBackend,
	uri: str,
	records: Iterable[Any],
	serializer: JsonLineSerializer,
	compression: Compression,
	mode: WriteMode,
) -> WriteResult:
	binary_mode = "ab" if mode == "a" else "wb"
	records_written = 0
	bytes_written = 0

	with backend.open(uri, binary_mode, atomic=mode == "w") as raw_stream:
		stream = _wrap_compression(raw_stream, StorageURI.parse(uri), compression)
		try:
			for record in records:
				line = serializer.dumps(record) + b"\n"
				stream.write(line)
				records_written += 1
				bytes_written += len(line)
		finally:
			if stream is not raw_stream:
				stream.close()

	return WriteResult(uri=uri, records_written=records_written, bytes_written=bytes_written, backend=backend.name)


def _wrap_compression(raw_stream: BinaryIO, uri: StorageURI, compression: Compression) -> BinaryIO:
	selected = _resolve_compression(uri, compression)
	if selected == "gzip":
		return gzip.GzipFile(fileobj=raw_stream, mode="wb")
	return raw_stream


def _wrap_decompression(raw_stream: BinaryIO, uri: StorageURI, compression: Compression) -> BinaryIO:
	selected = _resolve_compression(uri, compression)
	if selected == "gzip":
		return gzip.GzipFile(fileobj=raw_stream, mode="rb")
	return raw_stream


def _resolve_compression(uri: StorageURI, compression: Compression) -> Literal["none", "gzip"]:
	if compression == "infer":
		return "gzip" if ".gz" in uri.suffixes else "none"
	return compression


def _json_default(value: Any) -> Any:
	normalized = normalize_record(value)
	if normalized is not value:
		return normalized
	raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _is_write_mode(mode: str) -> bool:
	return any(flag in mode for flag in ("w", "a", "x", "+"))


def _is_replace_mode(mode: str) -> bool:
	return "w" in mode and "a" not in mode and "+" not in mode


def _import_fsspec() -> Any:
	try:
		import fsspec  # type: ignore[import-not-found]
	except ImportError as exc:
		raise MissingDependencyError(
			"Cloud/object storage requires fsspec plus the matching backend, such as `s3fs`, `gcsfs`, or `adlfs`."
		) from exc
	return fsspec


def _normalize_fsspec_target(uri: str, storage_options: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
	parsed_uri = StorageURI.parse(uri)
	options = dict(storage_options)
	scheme = "gs" if parsed_uri.scheme == "gcs" else parsed_uri.scheme

	if parsed_uri.scheme == "r2":
		scheme = "s3"
		endpoint_url = options.pop("endpoint_url", None) or os.environ.get("R2_ENDPOINT_URL")
		account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
		if endpoint_url is None and account_id:
			endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"
		if endpoint_url:
			client_kwargs = dict(options.get("client_kwargs", {}))
			client_kwargs.setdefault("endpoint_url", endpoint_url)
			options["client_kwargs"] = client_kwargs

	if parsed_uri.bucket is None:
		return uri, options

	key = f"/{parsed_uri.key}" if parsed_uri.key else ""
	return f"{scheme}://{parsed_uri.bucket}{key}", options


def _option_value(
	key: str,
	query: Mapping[str, Sequence[str]],
	options: Mapping[str, Any],
	default: Any,
) -> Any:
	if key in options:
		return options[key]
	values = query.get(key)
	if values:
		return values[0]
	return default


_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _quote_identifier(identifier: str) -> str:
	if not _IDENTIFIER_RE.match(identifier):
		raise StorageError(f"Invalid Postgres identifier: {identifier!r}")
	return f'"{identifier}"'


def _strip_storage_query_params(uri: str) -> str:
	parsed_uri = urlparse(uri)
	keep_query: list[tuple[str, str]] = []
	storage_keys = {"schema", "table", "dataset", "format"}
	for key, values in parse_qs(parsed_uri.query, keep_blank_values=True).items():
		if key not in storage_keys:
			keep_query.extend((key, value) for value in values)

	query = urlencode(keep_query)
	return urlunparse(parsed_uri._replace(query=query))


def _redact_connection_uri(uri: str) -> str:
	parsed_uri = urlparse(uri)
	if parsed_uri.password is None:
		return uri

	username = parsed_uri.username or ""
	hostname = parsed_uri.hostname or ""
	port = f":{parsed_uri.port}" if parsed_uri.port else ""
	netloc = f"{username}:***@{hostname}{port}"
	return urlunparse(parsed_uri._replace(netloc=netloc))


def _batched(records: Iterable[Any], batch_size: int) -> Iterator[list[Any]]:
	if batch_size <= 0:
		raise ValueError("batch_size must be greater than zero")

	batch: list[Any] = []
	for record in records:
		batch.append(record)
		if len(batch) >= batch_size:
			yield batch
			batch = []
	if batch:
		yield batch


register_backend(("file", ""), lambda storage_options=None: LocalStorageBackend())
register_backend(
	("s3", "r2", "gs", "gcs", "az", "abfs", "abfss"),
	lambda storage_options=None: FSSpecStorageBackend(storage_options=storage_options),
)
register_backend(
	("postgres", "postgresql", "pg"),
	lambda storage_options=None: PostgresStorageBackend(storage_options=storage_options),
)


__all__ = [
	"FSSpecStorageBackend",
	"JsonLineSerializer",
	"LocalStorageBackend",
	"MissingDependencyError",
	"PostgresStorageBackend",
	"StorageBackend",
	"StorageError",
	"StorageURI",
	"UnsupportedBackendError",
	"WriteResult",
	"get_backend",
	"normalize_record",
	"read_jsonl",
	"register_backend",
	"write_json",
	"write_jsonl",
]
