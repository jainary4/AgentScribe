from __future__ import annotations

import agentscribe.adapters.autogen as autogen


def test_autogen_package_reexports_public_api() -> None:
    assert set(autogen.__all__) == {
        "from_chat_history",
        "from_stream_events",
        "from_task_result",
        "messages_from_autogen_item",
    }
