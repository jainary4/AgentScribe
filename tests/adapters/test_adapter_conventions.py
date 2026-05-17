from __future__ import annotations

from pathlib import Path


def test_every_adapter_python_file_has_mirrored_unit_test() -> None:
    adapter_root = Path("agentscribe/adapters")
    test_root = Path("tests/adapters")

    missing: list[str] = []
    for source in sorted(adapter_root.rglob("*.py")):
        relative = source.relative_to(adapter_root)
        if relative.name == "__init__.py":
            expected = test_root / relative.parent / "test_init.py"
        else:
            expected = test_root / relative.parent / f"test_{relative.name}"
        if not expected.exists():
            missing.append(f"{source} -> {expected}")

    assert missing == []
