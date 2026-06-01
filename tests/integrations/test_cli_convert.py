from __future__ import annotations

import json

from click.testing import CliRunner

from agentscribe.cli import main

from .conftest import read_jsonl_file, write_jsonl_file


def test_cli_convert_jsonl_openai_chat_to_sharegpt_local_file(tmp_path) -> None:
    input_path = write_jsonl_file(
        tmp_path / "source.jsonl",
        [
            {
                "messages": [
                    {"role": "system", "content": "Be brief."},
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                ]
            }
        ],
    )
    output_path = tmp_path / "dataset.jsonl"

    result = CliRunner().invoke(
        main,
        ["convert", "jsonl", str(input_path), "--format", "sharegpt", "--output", str(output_path)],
    )

    assert result.exit_code == 0, result.output
    assert "Wrote 1 records" in result.output
    # ShareGPT keeps the system prompt in its own `system` field and out of the
    # `conversations` list (LLaMA-Factory convention) rather than duplicating it.
    assert read_jsonl_file(output_path) == [
        {
            "conversations": [
                {"from": "human", "value": "Hello"},
                {"from": "gpt", "value": "Hi"},
            ],
            "system": "Be brief.",
        }
    ]


def test_cli_convert_adapter_record_uses_registry_formatter_and_local_storage(tmp_path) -> None:
    input_path = tmp_path / "langgraph.json"
    input_path.write_text(
        json.dumps(
            {
                "state": {
                    "messages": [
                        {"role": "user", "content": "Summarize"},
                        {"role": "assistant", "content": "Summary"},
                    ],
                    "checkpoint": 3,
                },
                "config": {"configurable": {"thread_id": "thread-1"}},
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "langgraph-output.jsonl"

    result = CliRunner().invoke(
        main,
        ["convert", "langgraph", str(input_path), "--format", "prompt_completion", "--output", str(output_path)],
    )

    assert result.exit_code == 0, result.output
    assert read_jsonl_file(output_path) == [{"prompt": "Summarize", "completion": "Summary"}]


def test_cli_convert_stdin_to_stdout_keeps_status_on_stderr() -> None:
    payload = json.dumps({"prompt": "Q", "completion": "A"})

    result = CliRunner().invoke(
        main,
        ["convert", "prompt_completion", "-", "--format", "openai_chat", "--output", "-"],
        input=payload,
    )

    assert result.exit_code == 0, result.output
    output_lines = result.output.strip().splitlines()
    assert json.loads(output_lines[0]) == {
        "messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]
    }
    assert output_lines[1] == "Wrote 1 records to stdout"


def test_cli_convert_skip_invalid_continues_writing_valid_records(tmp_path) -> None:
    input_path = write_jsonl_file(
        tmp_path / "mixed.jsonl",
        [
            {"messages": [{"role": "user", "content": "valid"}]},
            {"messages": ["invalid"]},
            {"prompt": "also valid", "completion": "done"},
        ],
    )
    output_path = tmp_path / "mixed-output.jsonl"

    result = CliRunner().invoke(
        main,
        ["convert", "auto", str(input_path), "--output", str(output_path), "--skip-invalid"],
    )

    assert result.exit_code == 0, result.output
    assert "Skipping record 2" in result.output
    assert read_jsonl_file(output_path) == [
        {"messages": [{"role": "user", "content": "valid"}]},
        {"messages": [{"role": "user", "content": "also valid"}, {"role": "assistant", "content": "done"}]},
    ]
