# formatter.py

from typing import Literal, Optional
from agentscribe.core.canonical import CanonicalInteraction
from agentscribe.storage import write_jsonl
import json
import uuid


class Formatter:
    """Converts canonical interactions into fine-tuning dataset formats.

    Design rules (framework-agnostic):
      * Tool data is read from the STRUCTURED canonical fields
        (`tool_name`, `tool_args`, `tool_result`, `tool_call_id`) — never by
        re-parsing `content`. `content` is free text only.
      * The formatter trusts the chronological order of the messages it is
        given; it never reorders. (Correct ordering is the adapter's job.)
      * `tools` / function schemas are PASSED THROUGH from
        `interaction.metadata["tool_schemas"]` when present and omitted when
        absent — the formatter never invents a JSON schema from one example.
    """

    SUPPORTED_FORMATS = ["openai_chat", "alpaca", "sharegpt", "prompt_completion", "preference"]

    def __init__(self, format: Literal["openai_chat", "alpaca", "sharegpt", "prompt_completion", "preference"] = "openai_chat",*, alpaca_system_as_instruction: bool = False):
        self.format = format
        self.alpaca_system_as_instruction = alpaca_system_as_instruction

    # ------------------------------------------------------------------ #
    # Shared tool helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _args_to_json_str(args) -> str:
        """OpenAI requires `function.arguments` as a JSON *string*."""
        if args is None:
            return "{}"
        if isinstance(args, str):
            return args  # assume already a JSON string
        return json.dumps(args, ensure_ascii=False, default=str)

    @staticmethod
    def _resolve_tool_call(m: dict):
        """Return (name, args_dict) from structured fields, with a content fallback."""
        name = m.get("tool_name")
        args = m.get("tool_args")
        if name is None or args is None:
            content = m.get("content")
            if isinstance(content, str) and content.strip():
                try:
                    data = json.loads(content)
                    if isinstance(data, dict):
                        name = name or data.get("tool_name") or data.get("name")
                        if args is None:
                            args = data.get("tool_args") or data.get("arguments")
                except (json.JSONDecodeError, TypeError):
                    pass
        if name is None:
            name = "unknown_tool"
        if args is None:
            args = {}
        return name, args

    @staticmethod
    def _resolve_tool_result(m: dict) -> str:
        """Return the tool result as a string (tool content must be a string)."""
        result = m.get("tool_result")
        if result is None:
            result = m.get("content")
        if result is None:
            return ""
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False, default=str)

    @staticmethod
    def _tool_schemas(interaction: CanonicalInteraction):
        """Optional pass-through of OpenAI-style function schemas, if provided.

        Expected shape (set later by adapters that can read the tool registry):
            interaction.metadata["tool_schemas"] = [
                {"type": "function", "function": {"name": ..., "parameters": {...}}},
                ...
            ]
        """
        meta = getattr(interaction, "metadata", None) or {}
        schemas = meta.get("tool_schemas")
        if isinstance(schemas, list) and schemas:
            return schemas
        return None

    # ------------------------------------------------------------------ #
    # openai_chat
    # ------------------------------------------------------------------ #
    def _format_openai_chat(self, messages: list[dict]) -> dict:
        formatted: list[dict] = []
        pending: list[dict] = []  # tool_calls awaiting a result: {"id", "name"}

        def collect_tool_calls(start: int):
            """Group consecutive tool_call messages into one tool_calls array."""
            calls = []
            j = start
            while j < len(messages) and messages[j]["role"] == "tool_call":
                name, args = self._resolve_tool_call(messages[j])
                call_id = messages[j].get("tool_call_id") or f"call_{uuid.uuid4().hex[:8]}"
                calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": self._args_to_json_str(args)},
                })
                pending.append({"id": call_id, "name": name})
                j += 1
            return calls, j

        def resolve_result_id(m: dict) -> str:
            cid = m.get("tool_call_id")
            if cid:
                for k, p in enumerate(pending):
                    if p["id"] == cid:
                        pending.pop(k)
                        break
                return cid
            name = m.get("tool_name")
            for k, p in enumerate(pending):  # match the oldest unmatched call of same name
                if name is not None and p["name"] == name:
                    return pending.pop(k)["id"]
            if pending:                       # FIFO fallback
                return pending.pop(0)["id"]
            return "call_unknown"             # orphaned result (malformed input)

        i, n = 0, len(messages)
        while i < n:
            m = messages[i]
            role = m["role"]

            if role in ("system", "user"):
                formatted.append({"role": role, "content": m.get("content", "")})
                i += 1

            elif role == "assistant":
                text = m.get("content") or ""
                calls, j = collect_tool_calls(i + 1)
                if calls:
                    # Merge assistant text + its tool calls into one message.
                    formatted.append({
                        "role": "assistant",
                        "content": text if text.strip() else None,
                        "tool_calls": calls,
                    })
                    i = j
                else:
                    if text.strip():
                        formatted.append({"role": "assistant", "content": text})
                    i += 1

            elif role == "tool_call":
                calls, j = collect_tool_calls(i)
                formatted.append({"role": "assistant", "content": None, "tool_calls": calls})
                i = j

            elif role == "tool_response":
                formatted.append({
                    "role": "tool",
                    "tool_call_id": resolve_result_id(m),
                    "content": self._resolve_tool_result(m),
                })
                i += 1

            else:
                i += 1

        return {"messages": formatted}

    # ------------------------------------------------------------------ #
    # sharegpt  (LLaMA-Factory / Axolotl)
    # ------------------------------------------------------------------ #
    def _format_sharegpt(self, messages: list[dict]) -> dict:
        conversations = []
        system_content = ""

        for m in messages:
            role = m["role"]

            if role == "system":
                if not system_content:
                    system_content = m.get("content", "")
                continue  # system is a separate column, not a conversation turn

            if role == "user":
                conversations.append({"from": "human", "value": m.get("content", "")})

            elif role == "assistant":
                text = m.get("content") or ""
                if text.strip():  # skip empty assistant turns (would break alternation)
                    conversations.append({"from": "gpt", "value": text})

            elif role == "tool_call":
                name, args = self._resolve_tool_call(m)
                conversations.append({
                    "from": "function_call",
                    "value": json.dumps({"name": name, "arguments": args}, ensure_ascii=False, default=str),
                })

            elif role == "tool_response":
                conversations.append({"from": "observation", "value": self._resolve_tool_result(m)})

            else:
                conversations.append({"from": role, "value": m.get("content", "")})

        result = {"conversations": conversations}
        if system_content:
            result["system"] = system_content
        return result

    # ------------------------------------------------------------------ #
    # alpaca  (text-only SFT; tool turns are not representable here)
    # ------------------------------------------------------------------ #
    def _format_alpaca(self, messages: list[dict]) -> dict:
        
        system_msg = next((m for m in messages if m["role"] == "system"), None)
        user_msgs = [m for m in messages if m["role"] == "user"]
        assistant_msgs = [m for m in messages if m["role"] == "assistant" and (m.get("content") or "").strip()]
        last_user = user_msgs[-1]["content"] if user_msgs else ""
        last_output = assistant_msgs[-1]["content"] if assistant_msgs else ""

        if self.alpaca_system_as_instruction and system_msg:
            # Classic Stanford Alpaca: task (system prompt) -> instruction, user msg -> input.
            result = {"instruction": system_msg["content"], "input": last_user, "output": last_output}
        else:
            # LLaMA-Factory convention (default): user -> instruction; system prompt -> `system` field.
            result = {"instruction": last_user, "input": "", "output": last_output}
            if system_msg:
                result["system"] = system_msg["content"]

        pairs = list(zip(user_msgs, assistant_msgs))
        if len(pairs) > 1:
            result["history"] = [[u["content"], a["content"]] for u, a in pairs[:-1]]
        return result

    # ------------------------------------------------------------------ #
    # prompt_completion  (legacy, single-turn, text-only)
    # ------------------------------------------------------------------ #
    def _format_prompt_completion(self, messages: list[dict]) -> dict:
        user_msg = next((m for m in messages if m["role"] == "user"), None)
        assistant_msg = next(
            (m for m in reversed(messages) if m["role"] == "assistant" and (m.get("content") or "").strip()),
            None,
        )
        return {
            "prompt": user_msg["content"] if user_msg else "",
            "completion": assistant_msg["content"] if assistant_msg else "",
        }

    # ------------------------------------------------------------------ #
    # preference  (TRL DPO — conversational; `rejected` is judge-supplied)
    # ------------------------------------------------------------------ #
    def _format_preference(self, messages: list[dict]) -> dict:
        oa = self._format_openai_chat(messages)["messages"]
        chosen_idx = None
        for k in range(len(oa) - 1, -1, -1):
            if oa[k]["role"] == "assistant" and oa[k].get("content"):
                chosen_idx = k
                break
        if chosen_idx is None:
            return {"prompt": oa, "chosen": [], "rejected": [{"role": "assistant", "content": ""}]}
        return {
            "prompt": oa[:chosen_idx],          # full context incl. tool calls, as message objects
            "chosen": [oa[chosen_idx]],          # the real final assistant response
            "rejected": [{"role": "assistant", "content": ""}],  # filled by a human / LLM judge
        }

    # ------------------------------------------------------------------ #
    # Dispatch
    # ------------------------------------------------------------------ #
    def format_single(self, interaction: CanonicalInteraction) -> dict:
        """Convert one interaction into the target format."""
        messages = interaction.to_dict()["messages"]

        if self.format == "openai_chat":
            result = self._format_openai_chat(messages)
            schemas = self._tool_schemas(interaction)
            if schemas:
                result["tools"] = schemas  # OpenAI shape: [{"type":"function","function":{...}}]
            return result

        elif self.format == "sharegpt":
            result = self._format_sharegpt(messages)
            schemas = self._tool_schemas(interaction)
            if schemas:
                # LLaMA-Factory wants `tools` as a JSON *string* of unwrapped schemas.
                unwrapped = [s.get("function", s) for s in schemas]
                result["tools"] = json.dumps(unwrapped, ensure_ascii=False)
            return result

        elif self.format == "alpaca":
            return self._format_alpaca(messages)

        elif self.format == "prompt_completion":
            return self._format_prompt_completion(messages)

        elif self.format == "preference":
            return self._format_preference(messages)

        raise ValueError(f"Unsupported format: {self.format!r}. Choose from {self.SUPPORTED_FORMATS}.")

    def format_and_save(self, interactions: list[CanonicalInteraction], output_path: str):
        """Format all interactions and write to a local or cloud JSONL target."""
        result = write_jsonl(output_path, (self.format_single(interaction) for interaction in interactions))
        return result.records_written