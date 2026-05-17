from __future__ import annotations

from dataclasses import dataclass

from agentscribe.storage import LocalStorageBackend, StorageURI, get_backend, read_jsonl, write_json, write_jsonl


@dataclass
class Payload:
    name: str
    value: int


def test_local_storage_jsonl_round_trip_with_append_and_dataclass_serialization(tmp_path) -> None:
    output_path = tmp_path / "records.jsonl"

    first = write_jsonl(output_path, [{"id": 1}, Payload("two", 2)])
    second = write_jsonl(output_path, [{"id": 3}], mode="a")

    assert first.records_written == 2
    assert second.records_written == 1
    assert list(read_jsonl(output_path)) == [{"id": 1}, {"name": "two", "value": 2}, {"id": 3}]


def test_local_storage_infers_gzip_for_json_and_jsonl(tmp_path) -> None:
    json_path = tmp_path / "payload.json.gz"
    jsonl_path = tmp_path / "records.jsonl.gz"

    write_json(json_path, {"nested": {"ok": True}})
    write_jsonl(jsonl_path, [{"id": 1}, {"id": 2}])

    assert list(read_jsonl(jsonl_path)) == [{"id": 1}, {"id": 2}]
    assert json_path.exists()
    assert json_path.stat().st_size > 0


def test_storage_backend_resolution_matches_local_paths_and_file_uris(tmp_path) -> None:
    plain_path = tmp_path / "plain.jsonl"
    file_uri = f"file://{plain_path}"

    assert isinstance(get_backend(plain_path), LocalStorageBackend)
    assert isinstance(get_backend(file_uri), LocalStorageBackend)
    assert StorageURI.parse(plain_path).scheme == "file"
    assert StorageURI.parse(file_uri).path == str(plain_path)
