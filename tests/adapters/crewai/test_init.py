from __future__ import annotations

import agentscribe.adapters.crewai as crewai


def test_crewai_package_reexports_public_api() -> None:
    assert set(crewai.__all__) == {
        "CrewAIAdapter",
        "from_event",
        "from_kickoff_output",
        "from_llm_call_context",
        "from_tool_call_context",
    }
