from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Optional
import uuid

# --- Message role types ---
MessageRole = Literal["system", "user", "assistant", "tool_call", "tool_response"]

@dataclass
class CanonicalMessage:
    """One message in a conversation."""

    role: MessageRole
    content: str
    tool_name: Optional[str] = None
    tool_args: Optional[dict[str, Any]] = None
    tool_result: Optional[str] = None

    """msg = CanonicalMessage(
    role="user",
    content="What's 20% off $200?",
    )
    tool_name, tool_args, tool_result default to None"""

    #This method converts the message object(CanonicalMessage) into a plain Python dictionary, 
    # omitting optional fields that are None. It’s the bridge between our structured object and the JSON world

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary for serialization."""
        result: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
        }
        if self.tool_name is not None:
            result["tool_name"] = self.tool_name
        if self.tool_args is not None:
            result["tool_args"] = self.tool_args
        if self.tool_result is not None:
            result["tool_result"] = self.tool_result
        return result
    
    """Example :
    msg = CanonicalMessage(role="assistant", content="Hello")
    msg.to_dict()  
    # {'role': 'assistant', 'content': 'Hello'}"""

    @classmethod
    #This is a class method. It receives the class itself (cls) instead of an instance (self). 
    # It creates a new CanonicalMessage from a dictionary

    def from_dict(cls, data: dict[str, Any]) -> CanonicalMessage:
        """Create a CanonicalMessage from a dictionary."""
        return cls(
            role=data["role"],
            content=data["content"],
            tool_name=data.get("tool_name"),
            tool_args=data.get("tool_args"),
            tool_result=data.get("tool_result"),
        )
    """Example:
    data = {"role": "user", "content": "Hi", "tool_name": None}
    msg = CanonicalMessage.from_dict(data)
    # msg is now CanonicalMessage(role='user', content='Hi')"""

@dataclass
class CanonicalInteraction:
    """A complete agent interaction, ready for formatting."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: list[CanonicalMessage] = field(default_factory=list)
    source_framework: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    session_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    """We can’t use list or dict directly as default values 
    because Python would reuse the same mutable object across all instances. 
    field(default_factory=list) creates a fresh empty list each time. 
    Similarly, field(default_factory=lambda: str(uuid.uuid4())) generates a new UUID for every new interaction"""

    """interaction = CanonicalInteraction(
    source_framework="langgraph",
    session_id="sess_42",
    metadata={"agent": "math_solver"},
    )"""

    #interaction.id is something like "a1b2c3d4-..."
    # interaction.timestamp is the current time
    # interaction.messages is an empty list
    #interaction.messages.append(CanonicalMessage(role="system", content="You are a helpful assistant."))

    
    def to_dict(self) -> dict[str, Any]:

        """Convert the whole interaction to a dictionary."""
        """Example_output: {
        "id": "a1b2c3d4-...",
        "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"}],
        "source_framework": "langgraph",
        "timestamp": "2025-06-15T12:00:00+00:00",
        "session_id": "sess_42",
        "metadata": {"agent": "math_solver"}
        }"""

        return {
            "id": self.id,
            "messages": [m.to_dict() for m in self.messages],
            "source_framework": self.source_framework,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CanonicalInteraction:
        """Reconstruct an interaction from a dictionary."""
        messages = [CanonicalMessage.from_dict(m) for m in data["messages"]]
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            messages=messages,
            source_framework=data.get("source_framework", ""),
            timestamp=data.get("timestamp", ""),
            session_id=data.get("session_id"),
            metadata=data.get("metadata", {}),
        )

#How does a framework adapter builds the CanonicalInteraction instances:
"""# inside crewai.py adapter (pseudocode)
def on_llm_call(context):
    interaction = CanonicalInteraction(
        source_framework="crewai",
        session_id=context.session_id,
    )
    for turn in context.messages:
        interaction.messages.append(
            CanonicalMessage(role=turn["role"], content=turn["content"])
        )
    return interaction"""