from __future__ import annotations

import agentscribe.adapters as adapters


def test_adapter_package_exports_known_adapter_modules() -> None:
    assert set(adapters.__all__) == {
        "agentops",
        "agno",
        "atomic_agents",
        "autogen",
        "base",
        "crewai",
        "langgraph",
        "mcp",
        "mlflow",
        "opentelemetry",
        "openinference",
        "utils",
    }
