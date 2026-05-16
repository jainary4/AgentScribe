"""Model Context Protocol JSON-RPC adapter."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from agentscribe.core.canonical import CanonicalInteraction

from ..utils import compact_dict, get_nested, get_value, object_to_dict, parse_jsonish, tool_call_message, tool_response_message


def _request_id(message: Any) -> str | None:
    value = get_value(message, "id", default=None)
    return str(value) if value is not None else None


def _method(message: Any) -> str | None:
    method = get_value(message, "method", default=None)
    return str(method) if method is not None else None


def _params(message: Any) -> Any:
    return get_value(message, "params", default={})


def _result(message: Any) -> Any:
    return get_value(message, "result", default={})


def _session_id(*messages: Any) -> str | None:
    for message in messages:
        session_id = get_nested(message, "params", "_meta", "mcp.session.id", default=None)
        if session_id is None:
            session_id = get_nested(message, "result", "_meta", "mcp.session.id", default=None)
        if session_id is not None:
            return str(session_id)
    return None


def _content_from_tool_result(result: Any) -> Any:
    structured = get_value(result, "structuredContent", "structured_content", default=None)
    if structured is not None:
        return structured
    content = get_value(result, "content", default=None)
    if isinstance(content, list):
        extracted = []
        for item in content:
            text = get_value(item, "text", default=None)
            extracted.append(text if text is not None else object_to_dict(item) or str(item))
        return "\n".join(str(item) for item in extracted)
    return content if content is not None else result


def from_tool_call(request: Any, response: Any | None = None, *, metadata: Mapping[str, Any] | None = None) -> CanonicalInteraction:
    """Normalize an MCP tools/call request and optional JSON-RPC response."""

    params = _params(request)
    name = get_value(params, "name", "tool", default=None)
    arguments = parse_jsonish(get_value(params, "arguments", "args", default={}))
    request_id = _request_id(request)
    interaction = CanonicalInteraction(
        source_framework="mcp",
        session_id=_session_id(request, response),
        metadata={"method": "tools/call", **dict(metadata or {})},
    )
    interaction.messages.append(tool_call_message(str(name) if name is not None else None, arguments, tool_call_id=request_id))
    interaction.spans.append(
        {
            "kind": "mcp.request",
            "method": "tools/call",
            "jsonrpc.request.id": request_id,
            "request": object_to_dict(request) or request,
        }
    )
    if response is not None:
        result = _result(response)
        interaction.messages.append(
            tool_response_message(
                str(name) if name is not None else None,
                _content_from_tool_result(result),
                tool_call_id=request_id,
                metadata={"is_error": get_value(result, "isError", "is_error", default=False)},
            )
        )
        interaction.spans.append(
            {
                "kind": "mcp.response",
                "method": "tools/call",
                "jsonrpc.request.id": request_id,
                "response": object_to_dict(response) or response,
            }
        )
    return interaction


def from_tools_list(request: Any, response: Any | None = None, *, metadata: Mapping[str, Any] | None = None) -> CanonicalInteraction:
    """Normalize MCP tools/list traffic as instantiation/tool context."""

    interaction = CanonicalInteraction(
        source_framework="mcp",
        session_id=_session_id(request, response),
        metadata={"method": "tools/list", **dict(metadata or {})},
    )
    tools = get_value(_result(response), "tools", default=[]) if response is not None else []
    interaction.tools = [object_to_dict(tool) or {"name": str(tool)} for tool in tools]
    interaction.instantiation = compact_dict({"available_tools": interaction.tools})
    interaction.spans.append(
        {
            "kind": "mcp.request_response",
            "method": "tools/list",
            "jsonrpc.request.id": _request_id(request),
            "request": object_to_dict(request) or request,
            "response": object_to_dict(response) if response is not None else None,
        }
    )
    return interaction


def from_jsonrpc_pair(request: Any, response: Any | None = None, *, metadata: Mapping[str, Any] | None = None) -> CanonicalInteraction:
    """Normalize one MCP JSON-RPC request/response pair."""

    method = _method(request)
    if method == "tools/call":
        return from_tool_call(request, response, metadata=metadata)
    if method == "tools/list":
        return from_tools_list(request, response, metadata=metadata)
    interaction = CanonicalInteraction(
        source_framework="mcp",
        session_id=_session_id(request, response),
        metadata={"method": method, **dict(metadata or {})},
    )
    interaction.spans.append(
        {
            "kind": "mcp.request_response",
            "method": method,
            "jsonrpc.request.id": _request_id(request),
            "request": object_to_dict(request) or request,
            "response": object_to_dict(response) if response is not None else None,
        }
    )
    return interaction


def from_jsonrpc_messages(messages: Iterable[Any], *, metadata: Mapping[str, Any] | None = None) -> list[CanonicalInteraction]:
    """Pair JSON-RPC messages by id and normalize MCP interactions."""

    requests: dict[str, Any] = {}
    responses: dict[str, Any] = {}
    notifications: list[Any] = []
    for message in messages:
        message_id = _request_id(message)
        if _method(message) is not None:
            if message_id is not None:
                requests[message_id] = message
            else:
                notifications.append(message)
        elif message_id is not None:
            responses[message_id] = message

    interactions = [from_jsonrpc_pair(request, responses.get(request_id), metadata=metadata) for request_id, request in requests.items()]
    for notification in notifications:
        interactions.append(from_jsonrpc_pair(notification, None, metadata=metadata))
    return interactions


__all__ = [
    "from_jsonrpc_messages",
    "from_jsonrpc_pair",
    "from_tool_call",
    "from_tools_list",
]
