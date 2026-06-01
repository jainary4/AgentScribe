---
type: "query"
date: "2026-06-01T05:18:51.588437+00:00"
question: "Is the Agno Live test output a proper format for finetuning, distillation, and prompt improvement vs industry standards?"
contributor: "graphify"
source_nodes: ["Formatter", "CanonicalInteraction", "from_run_output", "tool_call_message", "AgnoAdapter"]
---

# Q: Is the Agno Live test output a proper format for finetuning, distillation, and prompt improvement vs industry standards?

## Answer

The Agno Live test uses Formatter(format='openai_chat'), which emits {messages:[{role,content}]} ONLY. The rich CanonicalInteraction layer (tool_name, tool_args, tool_result, tool_call_id, metadata, token_usage, model, provenance) is captured but DROPPED at the openai_chat formatting step. Output uses non-spec roles tool_call/tool_response, stringifies tool calls into content, emits empty assistant turns, and lacks tool_call_id linkage and loss weights. So the canonical SCHEMA matches/exceeds industry richness, but the emitted openai_chat format is lossy and NOT OpenAI-fine-tuning compliant. Distillation lacks logprobs/teacher metadata pass-through; prompt-improvement raw signal (status/error/metrics) exists in canonical but is stripped on export.

## Source Nodes

- Formatter
- CanonicalInteraction
- from_run_output
- tool_call_message
- AgnoAdapter