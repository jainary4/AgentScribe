"""Convert canonical interactions into industry-standard fine-tuning formats.

This module is the single source of truth for dataset serialization. Every
adapter and the CLI funnel through the same formatters so that exports are
spec-compliant and consumable for supervised fine-tuning, preference
optimization (DPO/ORPO/SimPO), distillation, and lossless archival.

Formats are registered by name and configured through :class:`FormatOptions`,
so new targets (e.g. vendor message schemas) can be added without touching the
dispatch logic. The core of every formatter operates on a plain list of
canonical message dicts plus an optional provenance mapping, which lets both
:class:`Formatter` (driven by a :class:`CanonicalInteraction`) and
:func:`format_messages` (driven by raw message dicts, as the CLI uses) share
exactly the same rendering path.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Optional

from agentscribe.core.canonical import CanonicalInteraction
from agentscribe.storage import write_jsonl


# --- Role handling -----------------------------------------------------------

CANONICAL_ROLES = {"system", "user", "assistant", "tool_call", "tool_response"}

# Map common framework-native role names onto canonical roles so the formatters
# behave the same whether they are fed a CanonicalInteraction or raw dicts.
_ROLE_ALIASES = {
    "ai": "assistant",
    "bot": "assistant",
    "gpt": "assistant",
    "human": "user",
    "function": "tool_response",
    "function_call": "tool_call",
    "observation": "tool_response",
    "tool": "tool_response",
}

# Canonical -> ShareGPT "from" values (LLaMA-Factory convention).
_SHAREGPT_FROM = {
    "system": "system",
    "user": "human",
    "assistant": "gpt",
    "tool_call": "function_call",
    "tool_response": "observation",
}

# Aux key that adapters tuck into tool_args to carry the tool_call_id / metadata
# (see adapters/utils/normalization.tool_call_message). Stripped from emitted
# function arguments but mined for the call id.
_AUX_KEY = "_agentscribe"


class FormatValidationError(ValueError):
    """Raised when ``strict`` is set and a produced record violates its spec."""


@dataclass(frozen=True)
class FormatOptions:
    """Configurable knobs shared by every formatter.

    All defaults keep strict training formats loader-clean; richer behaviour is
    opt-in so a single config can drive both training and distillation exports.
    """

    include_system: bool = True
    include_tools: bool = True            # emit a `tools` schema where supported
    include_metadata: bool = False        # embed provenance under `_agentscribe`
    drop_empty_assistant: bool = True     # drop assistant turns w/ no content & no tool_calls
    mask_non_final_assistant: bool = False  # OpenAI weight:0 on non-final assistant turns
    strict: bool = False                  # validate and raise on spec violation
    default_rejected: Optional[str] = None  # DPO fallback for `rejected`


# --- Small local helpers (kept here to avoid a core -> adapters import) -------

def _text(value: Any) -> str:
    """Coerce arbitrary content into message text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def _json_args(value: Any) -> str:
    """Serialize tool arguments to a JSON string (OpenAI/ShareGPT expect this)."""
    if isinstance(value, str):
        return value
    return json.dumps(value if value is not None else {}, ensure_ascii=False, default=str)


def _canonical_role(role: Any) -> str:
    role = str(role or "")
    return role if role in CANONICAL_ROLES else _ROLE_ALIASES.get(role, role)


def _strip_aux(args: Any) -> tuple[dict[str, Any], Optional[str]]:
    """Split a tool_args mapping into (clean_args, tool_call_id)."""
    if not isinstance(args, Mapping):
        return ({} if args is None else {"value": args}), None
    clean = {k: v for k, v in args.items() if k != _AUX_KEY}
    aux = args.get(_AUX_KEY) or {}
    call_id = aux.get("tool_call_id") if isinstance(aux, Mapping) else None
    return clean, (str(call_id) if call_id else None)


def _messages(interaction: CanonicalInteraction) -> list[dict[str, Any]]:
    """Return the interaction's messages as plain dicts (tool fields included)."""
    return list(interaction.to_dict()["messages"])


def _provenance(interaction: CanonicalInteraction) -> dict[str, Any]:
    """Collect interaction-level provenance, including dynamically-set attrs.

    Adapters set ``model`` / ``token_usage`` as attributes on the interaction
    (they are not declared dataclass fields and so never appear in ``to_dict``),
    which is why distillation provenance is read here via ``getattr``.
    """
    prov: dict[str, Any] = {}
    for attr in ("model", "token_usage", "source_framework", "session_id", "timestamp", "id"):
        value = getattr(interaction, attr, None)
        if value:
            prov[attr] = value
    metadata = getattr(interaction, "metadata", None)
    if metadata:
        prov["metadata"] = dict(metadata)
    return prov


def _split_turns(messages: list[dict[str, Any]]) -> tuple[Optional[str], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (system_text, user_messages, assistant_messages) by canonical role."""
    system = next((m for m in messages if _canonical_role(m.get("role")) == "system"), None)
    users = [m for m in messages if _canonical_role(m.get("role")) == "user"]
    assistants = [m for m in messages if _canonical_role(m.get("role")) == "assistant"]
    return (_text(system.get("content")) if system else None), users, assistants


def _tool_schemas(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Best-effort OpenAI ``tools`` schema list inferred from observed calls."""
    schemas: dict[str, dict[str, Any]] = {}
    for m in messages:
        if _canonical_role(m.get("role")) != "tool_call":
            continue
        name = m.get("tool_name") or "tool"
        clean, _ = _strip_aux(m.get("tool_args"))
        properties = {k: {"type": "string"} for k in clean} if isinstance(clean, dict) else {}
        schemas[name] = {
            "type": "function",
            "function": {
                "name": name,
                "parameters": {"type": "object", "properties": properties},
            },
        }
    return list(schemas.values())


# --- OpenAI message assembly (the spec-compliant core) -----------------------

def _assemble_openai(messages: list[dict[str, Any]], options: FormatOptions) -> list[dict[str, Any]]:
    """Build OpenAI-spec messages: structured assistant.tool_calls + tool role.

    Folds canonical ``tool_call`` messages into the preceding assistant turn's
    ``tool_calls`` array (synthesizing an empty-content assistant if needed) and
    converts ``tool_response`` into ``{"role":"tool","tool_call_id":...}``,
    generating and linking ids when the source omitted them.
    """
    out: list[dict[str, Any]] = []
    current_assistant: Optional[dict[str, Any]] = None
    pending: list[dict[str, str]] = []  # unmatched (id, name) for tool responses
    counter = 0

    def new_id(existing: Optional[str]) -> str:
        nonlocal counter
        if existing:
            return existing
        counter += 1
        return f"call_{counter}"

    for m in messages:
        role = _canonical_role(m.get("role"))
        content = _text(m.get("content"))

        if role == "system":
            if options.include_system:
                out.append({"role": "system", "content": content})
            current_assistant = None
        elif role == "user":
            out.append({"role": "user", "content": content})
            current_assistant = None
        elif role == "assistant":
            # If the message already carries structured tool_calls, trust them.
            msg: dict[str, Any] = {"role": "assistant", "content": content}
            if m.get("tool_calls"):
                msg["tool_calls"] = list(m["tool_calls"])
            out.append(msg)
            current_assistant = msg
        elif role == "tool_call":
            clean, call_id = _strip_aux(m.get("tool_args"))
            name = str(m.get("tool_name") or "tool")
            call_id = new_id(call_id)
            tool_call = {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": _json_args(clean)},
            }
            if current_assistant is None:
                current_assistant = {"role": "assistant", "content": "", "tool_calls": []}
                out.append(current_assistant)
            current_assistant.setdefault("tool_calls", [])
            current_assistant["tool_calls"].append(tool_call)
            pending.append({"id": call_id, "name": name})
        elif role == "tool_response":
            name = m.get("tool_name")
            result = m.get("tool_result")
            result = _text(result if result is not None else m.get("content"))
            # Prefer an explicit id, else match to a pending call by name, else
            # FIFO. A tool result with nothing to link to cannot be spec-valid,
            # so it is dropped rather than emitted with a synthetic id.
            _, explicit_id = _strip_aux(m.get("tool_args"))
            call_id = explicit_id or m.get("tool_call_id")
            if call_id:
                pending[:] = [p for p in pending if p["id"] != call_id]
            elif pending:
                match = next((p for p in pending if p["name"] == str(name)), pending[0])
                pending.remove(match)
                call_id = match["id"]
            if not call_id:
                current_assistant = None
                continue
            out.append({"role": "tool", "tool_call_id": call_id, "content": result})
            current_assistant = None
        else:
            # Unknown role: pass through conservatively as user content.
            out.append({"role": "user", "content": content})
            current_assistant = None

    if options.drop_empty_assistant:
        out = [
            m for m in out
            if not (m["role"] == "assistant" and not m["content"].strip() and not m.get("tool_calls"))
        ]

    if options.mask_non_final_assistant:
        assistant_indices = [i for i, m in enumerate(out) if m["role"] == "assistant"]
        last = assistant_indices[-1] if assistant_indices else None
        for i in assistant_indices:
            out[i]["weight"] = 1 if i == last else 0

    return out


# --- Formatter classes -------------------------------------------------------

class BaseFormatter(ABC):
    """Base class for all formatters. Subclasses implement :meth:`render`."""

    name: str = ""

    def __init__(self, options: Optional[FormatOptions] = None):
        self.options = options or FormatOptions()

    @abstractmethod
    def render(self, messages: list[dict[str, Any]], provenance: Mapping[str, Any]) -> Any:
        """Render a list of canonical message dicts into the target record."""

    def _maybe_metadata(self, record: dict[str, Any], provenance: Mapping[str, Any]) -> dict[str, Any]:
        if self.options.include_metadata and provenance:
            record[_AUX_KEY] = dict(provenance)
        return record

    def format_single(self, interaction: CanonicalInteraction) -> Any:
        record = self.render(_messages(interaction), _provenance(interaction))
        if self.options.strict and record is not None:
            self.validate(record)
        return record

    def validate(self, record: Any) -> None:  # pragma: no cover - overridden where meaningful
        """Validate a produced record against its spec; raise on violation."""


class OpenAIChatFormatter(BaseFormatter):
    """OpenAI / Azure supervised fine-tuning chat format."""

    name = "openai_chat"

    def render(self, messages, provenance):
        record: dict[str, Any] = {"messages": _assemble_openai(messages, self.options)}
        if self.options.include_tools:
            tools = _tool_schemas(messages)
            if tools:
                record["tools"] = tools
        return self._maybe_metadata(record, provenance)

    def validate(self, record):
        valid = {"system", "user", "assistant", "tool"}
        # Every tool message must reference a tool_call id issued by some
        # assistant message, otherwise OpenAI rejects the example.
        call_ids = {
            call["id"]
            for m in record["messages"]
            for call in (m.get("tool_calls") or [])
        }
        for m in record["messages"]:
            if m["role"] not in valid:
                raise FormatValidationError(f"invalid OpenAI role: {m['role']!r}")
            if m["role"] == "assistant" and not m.get("content", "").strip() and not m.get("tool_calls"):
                raise FormatValidationError("assistant message has neither content nor tool_calls")
            if m["role"] == "tool":
                call_id = m.get("tool_call_id")
                if not call_id:
                    raise FormatValidationError("tool message missing tool_call_id")
                if call_id not in call_ids:
                    raise FormatValidationError(
                        f"tool message tool_call_id {call_id!r} has no matching assistant tool_call"
                    )


class ShareGPTFormatter(BaseFormatter):
    """ShareGPT conversational format (LLaMA-Factory / Axolotl compatible)."""

    name = "sharegpt"

    def render(self, messages, provenance):
        conversations: list[dict[str, str]] = []
        system = ""
        for m in messages:
            role = _canonical_role(m.get("role"))
            if role == "system":
                if not system:
                    system = _text(m.get("content"))
                # System lives in its own field, not in `conversations`.
                continue
            if role == "tool_call":
                clean, _ = _strip_aux(m.get("tool_args"))
                value = json.dumps(
                    {"name": m.get("tool_name") or "tool", "arguments": clean},
                    ensure_ascii=False,
                    default=str,
                )
            elif role == "tool_response":
                value = _text(m.get("tool_result") if m.get("tool_result") is not None else m.get("content"))
            else:
                value = _text(m.get("content"))
            conversations.append({"from": _SHAREGPT_FROM.get(role, role), "value": value})

        record: dict[str, Any] = {"conversations": conversations}
        if self.options.include_system:
            record["system"] = system
        if self.options.include_tools:
            tools = _tool_schemas(messages)
            if tools:
                record["tools"] = json.dumps(tools, ensure_ascii=False)
        return self._maybe_metadata(record, provenance)


class AlpacaFormatter(BaseFormatter):
    """Alpaca instruction-tuning format with robust multi-turn history."""

    name = "alpaca"

    def render(self, messages, provenance):
        system, users, assistants = _split_turns(messages)
        instruction = _text(users[0].get("content")) if users else ""
        if system and self.options.include_system:
            instruction = f"{system}\n\n{instruction}".strip()

        record: dict[str, Any] = {
            "instruction": instruction,
            "input": "",
            "output": _text(assistants[-1].get("content")) if assistants else "",
        }
        # History = prior (user, assistant) pairs, tolerant of unequal counts.
        pairs = list(zip(users[1:], assistants[:-1]))
        if pairs:
            record["history"] = [[_text(u.get("content")), _text(a.get("content"))] for u, a in pairs]
        return self._maybe_metadata(record, provenance)


class PromptCompletionFormatter(BaseFormatter):
    """Legacy prompt/completion pair."""

    name = "prompt_completion"

    def render(self, messages, provenance):
        system, users, assistants = _split_turns(messages)
        prompt = _text(users[0].get("content")) if users else ""
        if system and self.options.include_system:
            prompt = f"{system}\n\n{prompt}".strip()
        record: dict[str, Any] = {
            "prompt": prompt,
            "completion": _text(assistants[-1].get("content")) if assistants else "",
        }
        return self._maybe_metadata(record, provenance)


class PreferenceFormatter(BaseFormatter):
    """Conversational preference pairs for DPO / ORPO / SimPO (TRL compatible)."""

    name = "preference"

    def render(self, messages, provenance):
        _, users, assistants = _split_turns(messages)
        if not assistants:
            if self.options.strict:
                raise FormatValidationError("preference record needs an assistant `chosen` response")
            return None

        rejected_text = self.options.default_rejected
        if rejected_text is None:
            metadata = provenance.get("metadata") if isinstance(provenance, Mapping) else None
            if isinstance(metadata, Mapping):
                rejected_text = metadata.get("rejected")
        if not rejected_text:
            if self.options.strict:
                raise FormatValidationError(
                    "preference record has no `rejected` response (set default_rejected or metadata['rejected'])"
                )
            return None

        record: dict[str, Any] = {
            "prompt": [{"role": "user", "content": _text(u.get("content"))} for u in users],
            "chosen": [{"role": "assistant", "content": _text(assistants[-1].get("content"))}],
            "rejected": [{"role": "assistant", "content": _text(rejected_text)}],
        }
        return self._maybe_metadata(record, provenance)


class ChatMLFormatter(BaseFormatter):
    """ChatML text format for generic HF/Axolotl SFT and teacher distillation."""

    name = "chatml"

    def render(self, messages, provenance):
        blocks: list[str] = []
        for m in messages:
            role = _canonical_role(m.get("role"))
            if role == "system" and not self.options.include_system:
                continue
            if role == "tool_call":
                clean, _ = _strip_aux(m.get("tool_args"))
                rendered = "assistant", json.dumps(
                    {"name": m.get("tool_name") or "tool", "arguments": clean}, ensure_ascii=False, default=str
                )
            elif role == "tool_response":
                rendered = "tool", _text(m.get("tool_result") if m.get("tool_result") is not None else m.get("content"))
            else:
                rendered = role, _text(m.get("content"))
            blocks.append(f"<|im_start|>{rendered[0]}\n{rendered[1]}<|im_end|>")
        return {"text": "\n".join(blocks)}


class CanonicalFormatter(BaseFormatter):
    """Lossless dump of the full canonical model (distillation / archival)."""

    name = "canonical"

    def render(self, messages, provenance):
        # Always carries everything, regardless of include_metadata.
        record: dict[str, Any] = {"messages": messages}
        record.update(provenance)
        return record

    def format_single(self, interaction: CanonicalInteraction) -> Any:
        record = dict(interaction.to_dict())
        prov = _provenance(interaction)
        for key in ("model", "token_usage"):
            if key in prov:
                record[key] = prov[key]
        return record


# --- Registry ----------------------------------------------------------------

_FORMATTERS: dict[str, type[BaseFormatter]] = {}


def register_format(name: str, formatter_cls: type[BaseFormatter]) -> None:
    """Register a formatter class under ``name`` (overrides any existing)."""
    _FORMATTERS[name] = formatter_cls


def available_formats() -> list[str]:
    """Return the list of registered format names."""
    return list(_FORMATTERS)


for _cls in (
    OpenAIChatFormatter,
    ShareGPTFormatter,
    AlpacaFormatter,
    PromptCompletionFormatter,
    PreferenceFormatter,
    ChatMLFormatter,
    CanonicalFormatter,
):
    register_format(_cls.name, _cls)


def _build_formatter(format_name: str, options: FormatOptions) -> BaseFormatter:
    try:
        formatter_cls = _FORMATTERS[format_name]
    except KeyError:
        raise ValueError(
            f"Unsupported format {format_name!r}. Supported formats: {', '.join(available_formats())}"
        ) from None
    return formatter_cls(options)


# --- Public facade -----------------------------------------------------------

class Formatter:
    """Converts canonical interactions into fine-tuning dataset formats."""

    SUPPORTED_FORMATS = list(_FORMATTERS)

    def __init__(
        self,
        format: str = "openai_chat",
        *,
        options: Optional[FormatOptions] = None,
        **option_overrides: Any,
    ):
        self.format = format
        base = options or FormatOptions()
        self.options = replace(base, **option_overrides) if option_overrides else base
        self._impl = _build_formatter(format, self.options)

    def format_single(self, interaction: CanonicalInteraction) -> Any:
        """Convert one interaction into the target format (may return None)."""
        return self._impl.format_single(interaction)

    def format_and_save(self, interactions: list[CanonicalInteraction], output_path: str):
        """Format all interactions and write to a local or cloud JSONL target."""
        records = (
            record
            for record in (self.format_single(interaction) for interaction in interactions)
            if record is not None
        )
        result = write_jsonl(output_path, records, format_name=self.format)
        return result.records_written


def format_messages(
    messages: Iterable[Mapping[str, Any]],
    format_name: str,
    *,
    provenance: Optional[Mapping[str, Any]] = None,
    options: Optional[FormatOptions] = None,
    **option_overrides: Any,
) -> Any:
    """Format a raw list of canonical message dicts (the CLI reuse hook).

    Shares the exact rendering path as :class:`Formatter` so the CLI ``convert``
    command and the adapters produce identical, spec-compliant output.
    """
    base = options or FormatOptions()
    resolved = replace(base, **option_overrides) if option_overrides else base
    formatter = _build_formatter(format_name, resolved)
    record = formatter.render([dict(m) for m in messages], provenance or {})
    if resolved.strict and record is not None:
        formatter.validate(record)
    return record
