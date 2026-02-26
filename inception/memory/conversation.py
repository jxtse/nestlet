"""
Conversation Memory.

Manages the history of user-agent interactions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from inception.provider.base import Message, MessageRole, ImageContent


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    # Tool usage in this turn
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_results: Optional[List[Dict[str, Any]]] = None
    # Multimodal content
    images: Optional[List[Dict[str, Any]]] = None
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_message(self) -> Message:
        """Convert to Message for LLM."""
        from inception.provider.base import ToolCall

        tool_calls_obj = None
        if self.tool_calls:
            tool_calls_obj = [
                ToolCall(
                    id=tc["id"],
                    name=tc["name"],
                    arguments=tc["arguments"],
                )
                for tc in self.tool_calls
            ]

        images_obj = None
        if self.images:
            images_obj = [
                ImageContent(
                    url=img.get("url"),
                    base64_data=img.get("base64_data"),
                    media_type=img.get("media_type", "image/png"),
                )
                for img in self.images
            ]

        return Message(
            role=self.role,
            content=self.content,
            images=images_obj,
            tool_calls=tool_calls_obj,
        )


class ConversationMemory:
    """
    Manages conversation history.

    Features:
    - Store and retrieve conversation turns
    - Convert to LLM message format
    - Summarization for long conversations
    - Context window management
    """

    def __init__(
        self,
        max_turns: int = 50,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize conversation memory.

        Args:
            max_turns: Maximum turns to keep in memory
            system_prompt: Optional system prompt
        """
        self._turns: List[ConversationTurn] = []
        self._max_turns = max_turns
        self._system_prompt = system_prompt
        self._summary: Optional[str] = None  # For summarized older context

    @property
    def system_prompt(self) -> Optional[str]:
        """Get the system prompt."""
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, prompt: str) -> None:
        """Set the system prompt."""
        self._system_prompt = prompt

    def add_user_message(
        self,
        content: str,
        images: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConversationTurn:
        """Add a user message, optionally with images."""
        turn = ConversationTurn(
            role=MessageRole.USER,
            content=content,
            images=images,
            metadata=metadata or {},
        )
        self._add_turn(turn)
        return turn

    def add_assistant_message(
        self,
        content: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConversationTurn:
        """Add an assistant message."""
        turn = ConversationTurn(
            role=MessageRole.ASSISTANT,
            content=content,
            tool_calls=tool_calls,
            metadata=metadata or {},
        )
        self._add_turn(turn)
        return turn

    def add_tool_result(
        self,
        tool_call_id: str,
        name: str,
        result: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConversationTurn:
        """Add a tool result."""
        turn = ConversationTurn(
            role=MessageRole.TOOL,
            content=result,
            tool_results=[{
                "tool_call_id": tool_call_id,
                "name": name,
                "result": result,
            }],
            metadata=metadata or {},
        )
        self._add_turn(turn)
        return turn

    def _add_turn(self, turn: ConversationTurn) -> None:
        """Add a turn to history, managing max size."""
        self._turns.append(turn)

        # Trim if exceeding max turns
        if len(self._turns) > self._max_turns:
            # Keep the most recent turns
            excess = len(self._turns) - self._max_turns
            # In future: summarize removed turns
            self._turns = self._turns[excess:]

    def get_messages(
        self,
        include_system: bool = True,
        max_messages: Optional[int] = None,
    ) -> List[Message]:
        """
        Get messages for LLM consumption.

        Args:
            include_system: Whether to include system prompt
            max_messages: Maximum messages to return

        Returns:
            List of Message objects
        """
        messages = []

        # Add system prompt
        if include_system and self._system_prompt:
            messages.append(Message.system(self._system_prompt))

        # Add summary if available
        if self._summary:
            messages.append(Message.system(
                f"Summary of earlier conversation:\n{self._summary}"
            ))

        # Add conversation turns
        turns = self._turns
        if max_messages:
            turns = turns[-max_messages:]

        for turn in turns:
            if turn.role == MessageRole.TOOL and turn.tool_results:
                # Add tool result messages
                for tr in turn.tool_results:
                    messages.append(Message.tool(
                        content=tr["result"],
                        tool_call_id=tr["tool_call_id"],
                        name=tr["name"],
                    ))
            else:
                messages.append(turn.to_message())

        return messages

    def get_last_user_message(self) -> Optional[ConversationTurn]:
        """Get the most recent user message."""
        for turn in reversed(self._turns):
            if turn.role == MessageRole.USER:
                return turn
        return None

    def get_last_assistant_message(self) -> Optional[ConversationTurn]:
        """Get the most recent assistant message."""
        for turn in reversed(self._turns):
            if turn.role == MessageRole.ASSISTANT:
                return turn
        return None

    def get_turn_count(self) -> int:
        """Get the number of turns."""
        return len(self._turns)

    def get_recent_turns(self, n: int = 5) -> List[ConversationTurn]:
        """Get the n most recent turns."""
        return self._turns[-n:] if self._turns else []

    def search_turns(
        self,
        query: str,
        role: Optional[MessageRole] = None,
    ) -> List[ConversationTurn]:
        """
        Search turns containing a query string.

        Args:
            query: String to search for
            role: Optional role filter

        Returns:
            Matching turns
        """
        results = []
        query_lower = query.lower()

        for turn in self._turns:
            if role and turn.role != role:
                continue
            if query_lower in turn.content.lower():
                results.append(turn)

        return results

    def summarize(self, provider: Any) -> str:
        """
        Summarize older conversation turns.

        This would use the LLM to create a summary.
        For now, returns a simple concatenation.

        Args:
            provider: LLM provider for summarization

        Returns:
            Summary string
        """
        # Simple implementation - can be enhanced with LLM summarization
        if len(self._turns) <= 10:
            return ""

        old_turns = self._turns[:-10]
        summary_parts = []

        for turn in old_turns:
            role_name = turn.role.value.capitalize()
            content_preview = turn.content[:100]
            if len(turn.content) > 100:
                content_preview += "..."
            summary_parts.append(f"- {role_name}: {content_preview}")

        return "\n".join(summary_parts)

    def clear(self) -> None:
        """Clear all conversation history."""
        self._turns.clear()
        self._summary = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "system_prompt": self._system_prompt,
            "summary": self._summary,
            "turns": [
                {
                    "role": turn.role.value,
                    "content": turn.content,
                    "timestamp": turn.timestamp.isoformat(),
                    "tool_calls": turn.tool_calls,
                    "tool_results": turn.tool_results,
                    "images": turn.images,
                    "metadata": turn.metadata,
                }
                for turn in self._turns
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ConversationMemory:
        """Create from dictionary."""
        memory = cls(
            system_prompt=data.get("system_prompt"),
        )
        memory._summary = data.get("summary")

        for turn_data in data.get("turns", []):
            turn = ConversationTurn(
                role=MessageRole(turn_data["role"]),
                content=turn_data["content"],
                timestamp=datetime.fromisoformat(turn_data["timestamp"]),
                tool_calls=turn_data.get("tool_calls"),
                tool_results=turn_data.get("tool_results"),
                images=turn_data.get("images"),
                metadata=turn_data.get("metadata", {}),
            )
            memory._turns.append(turn)

        return memory
