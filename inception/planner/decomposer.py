"""
Task Decomposition.

Breaks down complex tasks into smaller, manageable subtasks.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from inception.provider.base import BaseProvider

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class SubTask:
    """A subtask within a larger task."""
    id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    # Execution details
    suggested_approach: Optional[str] = None
    suggested_tools: List[str] = field(default_factory=list)
    # Results
    result: Optional[Any] = None
    error: Optional[str] = None

    def is_ready(self, completed_tasks: set[str]) -> bool:
        """Check if this subtask is ready to execute."""
        return all(dep in completed_tasks for dep in self.depends_on)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status.value,
            "depends_on": self.depends_on,
            "suggested_approach": self.suggested_approach,
            "suggested_tools": self.suggested_tools,
        }


@dataclass
class Task:
    """A task to be executed."""
    description: str
    subtasks: List[SubTask] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    # Context
    context: Dict[str, Any] = field(default_factory=dict)
    # Metadata
    complexity: float = 0.5
    estimated_steps: int = 1

    def get_next_subtask(self) -> Optional[SubTask]:
        """Get the next subtask ready for execution."""
        completed = {
            st.id for st in self.subtasks
            if st.status == TaskStatus.COMPLETED
        }

        for subtask in self.subtasks:
            if subtask.status == TaskStatus.PENDING and subtask.is_ready(completed):
                return subtask

        return None

    def get_progress(self) -> float:
        """Get task progress (0-1)."""
        if not self.subtasks:
            return 0.0

        completed = sum(
            1 for st in self.subtasks
            if st.status == TaskStatus.COMPLETED
        )
        return completed / len(self.subtasks)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "description": self.description,
            "status": self.status.value,
            "subtasks": [st.to_dict() for st in self.subtasks],
            "complexity": self.complexity,
            "progress": self.get_progress(),
        }


DECOMPOSITION_PROMPT = """Analyze the following task and break it down into subtasks.

Task: {task}

Context (if any): {context}

Please decompose this task into concrete, actionable subtasks. For each subtask:
1. Provide a clear description
2. Identify dependencies on other subtasks
3. Suggest an approach (neural reasoning, code execution, or specific tool)
4. Estimate complexity

Respond with a JSON object in this format:
{{
    "analysis": "Brief analysis of the overall task",
    "complexity": 0.5,  // 0-1 scale
    "subtasks": [
        {{
            "id": "task_1",
            "description": "Description of what to do",
            "depends_on": [],  // IDs of prerequisite subtasks
            "approach": "code_execution|neural_reasoning|tool:tool_name",
            "tools": ["tool1", "tool2"]  // Suggested tools
        }}
    ]
}}

Important:
- Make subtasks atomic and independent where possible
- Order subtasks logically
- Be specific about what each subtask should accomplish
- Identify clear dependencies
"""


class TaskDecomposer:
    """
    Decomposes complex tasks into subtasks.

    Uses LLM to analyze tasks and create execution plans.
    """

    def __init__(self, provider: "BaseProvider"):
        """
        Initialize the decomposer.

        Args:
            provider: LLM provider for task analysis
        """
        self._provider = provider

    async def decompose(
        self,
        task_description: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Task:
        """
        Decompose a task into subtasks.

        Args:
            task_description: Description of the task
            context: Optional context information

        Returns:
            Task with subtasks
        """
        from inception.provider.base import Message

        # Format prompt
        prompt = DECOMPOSITION_PROMPT.format(
            task=task_description,
            context=json.dumps(context) if context else "None",
        )

        # Call LLM
        messages = [
            Message.system(
                "You are a task planning assistant. Always respond with valid JSON."
            ),
            Message.user(prompt),
        ]

        response = await self._provider.complete(
            messages=messages,
            temperature=0.3,  # Lower temperature for consistent structure
        )

        # Parse response
        try:
            result = json.loads(response.content)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re
            json_match = re.search(r"\{.*\}", response.content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                logger.warning("Failed to parse decomposition response, creating single task")
                result = {
                    "analysis": "Could not decompose task",
                    "complexity": 0.5,
                    "subtasks": [{
                        "id": "task_1",
                        "description": task_description,
                        "depends_on": [],
                        "approach": "neural_reasoning",
                        "tools": [],
                    }],
                }

        # Create Task object
        subtasks = []
        for i, st_data in enumerate(result.get("subtasks", [])):
            subtask = SubTask(
                id=st_data.get("id", f"task_{i+1}"),
                description=st_data.get("description", ""),
                depends_on=st_data.get("depends_on", []),
                suggested_approach=st_data.get("approach"),
                suggested_tools=st_data.get("tools", []),
            )
            subtasks.append(subtask)

        task = Task(
            description=task_description,
            subtasks=subtasks,
            context=context or {},
            complexity=result.get("complexity", 0.5),
            estimated_steps=len(subtasks),
        )

        return task

    async def refine_subtask(
        self,
        subtask: SubTask,
        feedback: str,
    ) -> SubTask:
        """
        Refine a subtask based on feedback.

        Args:
            subtask: Subtask to refine
            feedback: Feedback about why refinement is needed

        Returns:
            Refined subtask
        """
        from inception.provider.base import Message

        prompt = f"""Refine the following subtask based on the feedback provided.

Subtask: {subtask.description}
Current approach: {subtask.suggested_approach}
Suggested tools: {subtask.suggested_tools}

Feedback: {feedback}

Provide a refined version as JSON:
{{
    "description": "Refined description",
    "approach": "Refined approach",
    "tools": ["tool1", "tool2"]
}}
"""

        messages = [
            Message.system("You are a task planning assistant. Respond with valid JSON."),
            Message.user(prompt),
        ]

        response = await self._provider.complete(
            messages=messages,
            temperature=0.3,
        )

        try:
            result = json.loads(response.content)
            subtask.description = result.get("description", subtask.description)
            subtask.suggested_approach = result.get("approach", subtask.suggested_approach)
            subtask.suggested_tools = result.get("tools", subtask.suggested_tools)
        except json.JSONDecodeError:
            logger.warning("Failed to parse refinement response")

        return subtask

    def is_task_simple(self, task_description: str) -> bool:
        """
        Check if a task is simple enough to execute directly.

        Simple tasks don't need decomposition.

        Args:
            task_description: Task description

        Returns:
            True if task is simple
        """
        # Heuristics for simple tasks
        words = task_description.split()

        # Very short tasks are usually simple
        if len(words) < 10:
            return True

        # Tasks without step indicators are often simple
        step_indicators = [
            "first", "then", "next", "after", "finally",
            "step", "steps", "1.", "2.", "3.",
        ]
        if not any(ind in task_description.lower() for ind in step_indicators):
            # Check for complexity indicators
            complexity_indicators = [
                "analyze", "compare", "process", "transform",
                "multiple", "several", "each", "all",
            ]
            if not any(ind in task_description.lower() for ind in complexity_indicators):
                return True

        return False
