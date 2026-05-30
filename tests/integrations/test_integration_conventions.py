from __future__ import annotations

from pathlib import Path


def test_integration_tests_live_in_dedicated_folder() -> None:
    # Find the integration test directory relative to this test file
    this_file = Path(__file__).resolve()
    integration_root = this_file.parent  # Because this file lives inside tests/integration

    assert integration_root.is_dir()
    assert sorted(path.name for path in integration_root.glob("test_*.py")) == [
        "test_cli_convert.py",
        "test_collectors.py",
        "test_integration_conventions.py",
        "test_storage_local.py",
    ]
