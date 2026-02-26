"""
Working Memory.

Manages the current task context and intermediate results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class MemoryItemType(str, Enum):
    """Types of items in working memory."""
    TASK = "task"
    SUBTASK = "subtask"
    OBSERVATION = "observation"
    RESULT = "result"
    VARIABLE = "variable"
    ARTIFACT = "artifact"
    DECISION = "decision"
    ERROR = "error"


@dataclass
class MemoryItem:
    """An item in working memory."""
    type: MemoryItemType
    content: Any
    name: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    # For linking related items
    parent_id: Optional[str] = None
    related_ids: List[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        """Generate a unique ID for this item."""
        return f"{self.type.value}_{self.created_at.timestamp()}"

    def is_expired(self) -> bool:
        """Check if this item has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at


class WorkingMemory:
    """
    Working memory for the current task.

    Features:
    - Store task decomposition
    - Track intermediate results
    - Manage variable references
    - Support item expiration
    """

    def __init__(self, max_items: int = 20):
        """
        Initialize working memory.

        Args:
            max_items: Maximum items to keep
        """
        self._items: Dict[str, MemoryItem] = {}
        self._max_items = max_items
        self._current_task: Optional[str] = None
        self._task_stack: List[str] = []

    def add(
        self,
        item_type: MemoryItemType,
        content: Any,
        name: Optional[str] = None,
        ttl_seconds: Optional[float] = None,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add an item to working memory.

        Args:
            item_type: Type of item
            content: Item content
            name: Optional name for the item
            ttl_seconds: Time-to-live in seconds
            parent_id: Optional parent item ID
            metadata: Optional metadata

        Returns:
            Item ID
        """
        expires_at = None
        if ttl_seconds:
            from datetime import timedelta
            expires_at = datetime.now() + timedelta(seconds=ttl_seconds)

        item = MemoryItem(
            type=item_type,
            content=content,
            name=name,
            expires_at=expires_at,
            parent_id=parent_id,
            metadata=metadata or {},
        )

        self._items[item.id] = item

        # Clean up if over limit
        self._cleanup()

        return item.id

    def get(self, item_id: str) -> Optional[MemoryItem]:
        """Get an item by ID."""
        item = self._items.get(item_id)
        if item and item.is_expired():
            del self._items[item_id]
            return None
        return item

    def get_by_name(self, name: str) -> Optional[MemoryItem]:
        """Get an item by name."""
        for item in self._items.values():
            if item.name == name and not item.is_expired():
                return item
        return None

    def get_by_type(self, item_type: MemoryItemType) -> List[MemoryItem]:
        """Get all items of a specific type."""
        return [
            item for item in self._items.values()
            if item.type == item_type and not item.is_expired()
        ]

    def remove(self, item_id: str) -> bool:
        """Remove an item by ID."""
        if item_id in self._items:
            del self._items[item_id]
            return True
        return False

    def update(
        self,
        item_id: str,
        content: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Update an existing item.

        Args:
            item_id: Item ID to update
            content: New content (if provided)
            metadata: Metadata to merge

        Returns:
            True if item was found and updated
        """
        item = self._items.get(item_id)
        if not item:
            return False

        if content is not None:
            item.content = content

        if metadata:
            item.metadata.update(metadata)

        return True

    def link_items(self, item_id: str, related_id: str) -> bool:
        """Link two items as related."""
        item = self._items.get(item_id)
        if not item:
            return False

        if related_id not in item.related_ids:
            item.related_ids.append(related_id)

        # Also add reverse link
        related = self._items.get(related_id)
        if related and item_id not in related.related_ids:
            related.related_ids.append(item_id)

        return True

    # Task management

    def set_current_task(self, task_description: str) -> str:
        """
        Set the current task.

        Args:
            task_description: Task description

        Returns:
            Task item ID
        """
        task_id = self.add(
            item_type=MemoryItemType.TASK,
            content=task_description,
            name="current_task",
        )
        self._current_task = task_id
        return task_id

    def get_current_task(self) -> Optional[MemoryItem]:
        """Get the current task."""
        if self._current_task:
            return self.get(self._current_task)
        return None

    def push_subtask(self, subtask_description: str) -> str:
        """
        Push a subtask onto the task stack.

        Args:
            subtask_description: Subtask description

        Returns:
            Subtask item ID
        """
        if self._current_task:
            self._task_stack.append(self._current_task)

        subtask_id = self.add(
            item_type=MemoryItemType.SUBTASK,
            content=subtask_description,
            parent_id=self._current_task,
        )
        self._current_task = subtask_id
        return subtask_id

    def pop_subtask(self) -> Optional[str]:
        """
        Pop the current subtask and return to parent.

        Returns:
            Parent task ID or None
        """
        if self._task_stack:
            self._current_task = self._task_stack.pop()
            return self._current_task
        return None

    # Observation and result tracking

    def add_observation(
        self,
        content: str,
        name: Optional[str] = None,
    ) -> str:
        """Add an observation."""
        return self.add(
            item_type=MemoryItemType.OBSERVATION,
            content=content,
            name=name,
            parent_id=self._current_task,
        )

    def add_result(
        self,
        content: Any,
        name: Optional[str] = None,
    ) -> str:
        """Add a result."""
        return self.add(
            item_type=MemoryItemType.RESULT,
            content=content,
            name=name,
            parent_id=self._current_task,
        )

    def add_decision(
        self,
        decision: str,
        reasoning: Optional[str] = None,
    ) -> str:
        """Add a decision."""
        return self.add(
            item_type=MemoryItemType.DECISION,
            content=decision,
            metadata={"reasoning": reasoning} if reasoning else {},
            parent_id=self._current_task,
        )

    def add_error(
        self,
        error: str,
        error_type: Optional[str] = None,
    ) -> str:
        """Add an error."""
        return self.add(
            item_type=MemoryItemType.ERROR,
            content=error,
            metadata={"error_type": error_type} if error_type else {},
            parent_id=self._current_task,
        )

    # Context for agent

    def get_context_summary(self) -> Dict[str, Any]:
        """
        Get a summary of working memory for the agent.

        Returns:
            Context summary dictionary
        """
        self._cleanup()  # Remove expired items first

        current_task = self.get_current_task()

        return {
            "current_task": current_task.content if current_task else None,
            "task_stack_depth": len(self._task_stack),
            "observations": [
                {"name": item.name, "content": item.content}
                for item in self.get_by_type(MemoryItemType.OBSERVATION)
            ],
            "results": [
                {"name": item.name, "content": str(item.content)[:200]}
                for item in self.get_by_type(MemoryItemType.RESULT)
            ],
            "decisions": [
                item.content
                for item in self.get_by_type(MemoryItemType.DECISION)
            ],
            "errors": [
                item.content
                for item in self.get_by_type(MemoryItemType.ERROR)
            ],
            "total_items": len(self._items),
        }

    def _cleanup(self) -> None:
        """Remove expired items and trim to max size."""
        # Remove expired
        expired = [
            item_id for item_id, item in self._items.items()
            if item.is_expired()
        ]
        for item_id in expired:
            del self._items[item_id]

        # Trim if over limit (remove oldest non-task items)
        if len(self._items) > self._max_items:
            # Sort by creation time
            sorted_items = sorted(
                self._items.items(),
                key=lambda x: x[1].created_at,
            )

            # Remove oldest items that aren't tasks
            to_remove = len(self._items) - self._max_items
            removed = 0
            for item_id, item in sorted_items:
                if item.type not in (MemoryItemType.TASK, MemoryItemType.SUBTASK):
                    del self._items[item_id]
                    removed += 1
                    if removed >= to_remove:
                        break

    def clear(self) -> None:
        """Clear all working memory."""
        self._items.clear()
        self._current_task = None
        self._task_stack.clear()
