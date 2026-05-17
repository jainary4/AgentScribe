from __future__ import annotations

import agentscribe.adapters.atomic_agents as atomic_agents


def test_atomic_agents_package_reexports_public_api() -> None:
    assert set(atomic_agents.__all__) == {
        "AtomicAgentsTraceCollector",
        "from_agent_response",
        "from_agent_run",
        "from_chat_history",
        "from_log_event",
    }
    assert callable(atomic_agents.from_agent_response)
