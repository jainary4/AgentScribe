# AgentScribe + MCP: Capture Guide

This guide explains how AgentScribe converts Model Context Protocol JSON-RPC traffic into canonical interactions for dataset generation.

---

## Supported Inputs

| Input shape | What it captures | Converter |
|---|---|---|
| `tools/call` request and response | Tool name, arguments, structured/text result, JSON-RPC id, session metadata | `from_tool_call()` |
| `tools/list` request and response | Available tool definitions and instantiation metadata | `from_tools_list()` |
| Request/response pair | Any JSON-RPC pair, with MCP tool calls promoted to messages | `from_jsonrpc_pair()` |
| Message list | Multiple JSON-RPC requests, responses, and notifications paired by id | `from_jsonrpc_messages()` |

MCP records become `CanonicalInteraction` objects with `source_framework="mcp"`. Tool calls are represented as `tool_call` messages, and tool results are represented as `tool_response` messages.

---

## Python Usage

```python
from agentscribe.adapters.mcp import from_jsonrpc_pair

interaction = from_jsonrpc_pair(
    {
        "jsonrpc": "2.0",
        "id": "call-1",
        "method": "tools/call",
        "params": {
            "name": "search",
            "arguments": {"query": "agent training data"},
        },
    },
    {
        "jsonrpc": "2.0",
        "id": "call-1",
        "result": {
            "content": [{"type": "text", "text": "Result text"}],
            "isError": False,
        },
    },
)
```

For logs containing many JSON-RPC messages, use `from_jsonrpc_messages(messages)` and let AgentScribe pair requests and responses by `id`.

---

## CLI Usage

```bash
agentscribe convert mcp ./mcp_messages.json --format openai_chat --output ./dataset.jsonl
```

Input records should contain either:

- `messages`: a list of JSON-RPC messages
- `request`: one JSON-RPC request, with optional `response`

---

## Canonical Mapping

| MCP concept | Canonical representation |
|---|---|
| `tools/call.params.name` | `CanonicalMessage.tool_name` |
| `tools/call.params.arguments` | `CanonicalMessage.tool_args` |
| `tools/call.result.structuredContent` or `content` | `CanonicalMessage.tool_result` |
| JSON-RPC `id` | Tool call id metadata |
| `_meta["mcp.session.id"]` | `CanonicalInteraction.session_id` |
| Raw request/response | `CanonicalInteraction.spans` |
