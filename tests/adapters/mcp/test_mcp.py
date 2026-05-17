from __future__ import annotations

from agentscribe.adapters.mcp import from_jsonrpc_messages, from_jsonrpc_pair, from_tool_call, from_tools_list


def test_from_tool_call_normalizes_request_and_structured_response() -> None:
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "search", "arguments": '{"q":"x"}', "_meta": {"mcp.session.id": "session-1"}},
    }
    response = {"id": 1, "result": {"structuredContent": {"answer": "y"}, "isError": False}}

    interaction = from_tool_call(request, response, metadata={"case": "call"})

    assert interaction.session_id == "session-1"
    assert interaction.metadata["method"] == "tools/call"
    assert interaction.metadata["case"] == "call"
    assert [message.role for message in interaction.messages] == ["tool_call", "tool_response"]
    assert interaction.messages[0].tool_args["q"] == "x"
    assert interaction.messages[1].content == '{"answer": "y"}'
    assert len(interaction.spans) == 2


def test_from_tool_call_handles_missing_response_boundary() -> None:
    interaction = from_tool_call({"id": "r1", "params": {"name": "noop"}})

    assert len(interaction.messages) == 1
    assert interaction.messages[0].tool_name == "noop"


def test_from_tool_call_extracts_text_content_list() -> None:
    interaction = from_tool_call(
        {"id": 2, "params": {"name": "read"}},
        {"id": 2, "result": {"content": [{"text": "a"}, {"text": "b"}]}},
    )

    assert interaction.messages[1].content == "a\nb"


def test_from_tools_list_records_available_tools() -> None:
    interaction = from_tools_list(
        {"id": "list", "method": "tools/list"},
        {"id": "list", "result": {"tools": [{"name": "search"}, "raw"]}},
    )

    assert interaction.tools == [{"name": "search"}, {"name": "raw"}]
    assert interaction.instantiation["available_tools"] == interaction.tools
    assert interaction.spans[0]["method"] == "tools/list"


def test_from_jsonrpc_pair_routes_known_methods_and_preserves_unknown_methods() -> None:
    assert from_jsonrpc_pair({"id": 1, "method": "tools/call", "params": {"name": "t"}}).metadata["method"] == "tools/call"
    assert from_jsonrpc_pair({"id": 2, "method": "tools/list"}).metadata["method"] == "tools/list"

    unknown = from_jsonrpc_pair({"id": 3, "method": "ping"}, {"id": 3, "result": "pong"})
    assert unknown.metadata["method"] == "ping"
    assert unknown.messages == []
    assert unknown.spans[0]["response"] == {"id": 3, "result": "pong"}


def test_from_jsonrpc_messages_pairs_requests_responses_and_notifications() -> None:
    interactions = from_jsonrpc_messages(
        [
            {"id": "1", "method": "tools/call", "params": {"name": "first"}},
            {"method": "notifications/progress", "params": {"pct": 50}},
            {"id": "1", "result": {"content": [{"text": "ok"}]}},
        ],
        metadata={"batch": True},
    )

    assert [item.metadata["method"] for item in interactions] == ["tools/call", "notifications/progress"]
    assert interactions[0].messages[1].content == "ok"
    assert all(item.metadata["batch"] is True for item in interactions)


def test_from_jsonrpc_messages_accepts_empty_iterable_boundary() -> None:
    assert from_jsonrpc_messages([]) == []
