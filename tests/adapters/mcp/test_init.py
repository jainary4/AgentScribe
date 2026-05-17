from __future__ import annotations

import agentscribe.adapters.mcp as mcp


def test_mcp_package_reexports_public_api() -> None:
    assert set(mcp.__all__) == {
        "from_jsonrpc_messages",
        "from_jsonrpc_pair",
        "from_tool_call",
        "from_tools_list",
    }
