from __future__ import annotations

import agentscribe.adapters.agentops as agentops


def test_agentops_package_reexports_public_converters() -> None:
    assert set(agentops.__all__) == {"from_events", "from_trace"}
    assert callable(agentops.from_events)
    assert callable(agentops.from_trace)
