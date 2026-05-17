from __future__ import annotations

import agentscribe.adapters.agno as agno


def test_agno_package_reexports_public_api_and_alias() -> None:
    assert set(agno.__all__) == {
        "AgnoAdapter",
        "AgnoHookAdapter",
        "AgnoTraceCollector",
        "from_event_stream",
        "from_run_output",
        "from_session",
        "from_trace",
        "parse_agno_run_output",
    }
    assert agno.AgnoHookAdapter is agno.AgnoAdapter
