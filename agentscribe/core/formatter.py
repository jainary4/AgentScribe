# formatter.py

import json
from pathlib import Path
from typing import Literal
from agentscribe.core.canonical import CanonicalInteraction

class Formatter:
    """Converts canonical interactions into fine-tuning dataset formats."""

    SUPPORTED_FORMATS = ["openai_chat", "alpaca", "sharegpt", "prompt_completion", "preference"]

    def __init__(self, format: Literal["openai_chat", "alpaca", "sharegpt", "prompt_completion", "preference"] = "openai_chat"):
        self.format = format

    def format_single(self, interaction: CanonicalInteraction) -> dict:
        """Convert one interaction into the target format."""
        messages = interaction.to_dict()["messages"]

        if self.format == "openai_chat":
            # The blog's format: {"messages": [{"role": ..., "content": ...}, ...]}
            return {"messages": [{"role": m["role"], "content": m["content"]} for m in messages]}

        elif self.format == "alpaca":
            # instruction = first user message (+ optional system)
            # output = last assistant message
            system_msg = next((m for m in messages if m["role"] == "system"), None)
            user_msgs = [m for m in messages if m["role"] == "user"]
            assistant_msgs = [m for m in messages if m["role"] == "assistant"]

            result = {
                "instruction": user_msgs[0]["content"] if user_msgs else "",
                "input": "",
                "output": assistant_msgs[-1]["content"] if assistant_msgs else "",
            }
            if system_msg:
                # Move system content to instruction prefix
                result["instruction"] = system_msg["content"] + "\n\n" + result["instruction"]
            # Multi-turn history
            if len(user_msgs) > 1:
                result["history"] = [
                    [user_msgs[i]["content"], assistant_msgs[i]["content"]]
                    for i in range(1, len(user_msgs))
                ]
            return result

        elif self.format == "sharegpt":
            # conversations: [{"from": "human"/"gpt"/"function_call"/"observation", "value": ...}, ...]
            role_map = {
                "system": "system",
                "user": "human",
                "assistant": "gpt",
                "tool_call": "function_call",
                "tool_response": "observation",
            }
            conversations = []
            for m in messages:
                from_role = role_map.get(m["role"], m["role"])
                conversations.append({"from": from_role, "value": m["content"]})
            return {
                "conversations": conversations,
                "system": next((m["content"] for m in messages if m["role"] == "system"), ""),
            }

        elif self.format == "prompt_completion":
            # Simplest: first user message → prompt, last assistant message → completion
            user_msg = next((m for m in messages if m["role"] == "user"), None)
            assistant_msg = next((m for m in reversed(messages) if m["role"] == "assistant"), None)
            return {
                "prompt": user_msg["content"] if user_msg else "",
                "completion": assistant_msg["content"] if assistant_msg else "",
            }

        elif self.format == "preference":
            # For DPO: chosen = assistant response, rejected = placeholder (user provides)
            user_content = " ".join([m["content"] for m in messages if m["role"] == "user"])
            assistant_content = " ".join([m["content"] for m in messages if m["role"] == "assistant"])
            return {
                "prompt": [{"role": "user", "content": user_content}],
                "chosen": [{"role": "assistant", "content": assistant_content}],
                "rejected": [{"role": "assistant", "content": ""}],  # user fills in the bad response
            }

    def format_and_save(self, interactions: list[CanonicalInteraction], output_path: str):
        """Format all interactions and write to a JSONL file."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, "w", encoding="utf-8") as f:
            for interaction in interactions:
                formatted = self.format_single(interaction)
                f.write(json.dumps(formatted, ensure_ascii=False) + "\n")

        return len(interactions)