"""
Agent base classes.

Defines the interface for all agents in the Inception system.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class ActionType(str, Enum):
    """Types of actions an agent can take."""
    TOOL_CALL = "tool_call"  # Execute a tool
    CODE_EXEC = "code_exec"  # Execute code directly
    LLM_CALL = "llm_call"  # Make an LLM call
    CREATE_TOOL = "create_tool"  # Create a new tool
    RESPOND = "respond"  # Respond to user
    DELEGATE = "delegate"  # Delegate to another agent
    WAIT = "wait"  # Wait for external input


@dataclass
class Context:
    """
    Context for agent reasoning and action.

    Contains all information the agent needs to make decisions.
    """
    # Current user request
    user_input: str
    # Conversation history summary
    conversation_summary: Optional[str] = None
    # Working memory state
    working_memory: Optional[Dict[str, Any]] = None
    # Available tools
    available_tools: List[str] = field(default_factory=list)
    # Execution state (variables, artifacts)
    execution_state: Optional[Dict[str, Any]] = None
    # Task decomposition
    current_task: Optional[str] = None
    subtasks: List[str] = field(default_factory=list)
    completed_subtasks: List[str] = field(default_factory=list)
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_prompt_context(self) -> str:
        """Format context for inclusion in LLM prompt."""
        parts = []

        if self.current_task:
            parts.append(f"Current Task: {self.current_task}")

        if self.subtasks:
            parts.append(f"Remaining Subtasks: {', '.join(self.subtasks)}")

        if self.completed_subtasks:
            parts.append(f"Completed: {', '.join(self.completed_subtasks)}")

        if self.working_memory:
            wm = self.working_memory
            if wm.get("observations"):
                parts.append("Observations:")
                for obs in wm["observations"]:
                    parts.append(f"  - {obs.get('name', 'unnamed')}: {obs['content']}")
            if wm.get("results"):
                parts.append("Results:")
                for res in wm["results"]:
                    parts.append(f"  - {res.get('name', 'unnamed')}: {res['content']}")

        if self.execution_state:
            state = self.execution_state
            if state.get("variables"):
                vars_info = [
                    f"{name}: {info['type']}"
                    for name, info in state["variables"].items()
                ]
                parts.append(f"Variables: {', '.join(vars_info)}")

        if self.available_tools:
            parts.append(f"Available Tools: {', '.join(self.available_tools)}")

        return "\n".join(parts)


@dataclass
class ThinkResult:
    """Result of agent thinking/reasoning."""
    # Reasoning trace
    reasoning: str
    # Decided action
    action: Action
    # Confidence in this decision (0-1)
    confidence: float = 1.0
    # Alternative actions considered
    alternatives: List[Action] = field(default_factory=list)
    # Whether more thinking is needed
    needs_more_thought: bool = False


@dataclass
class Action:
    """An action to be executed."""
    type: ActionType
    # Action-specific data
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    code: Optional[str] = None
    response: Optional[str] = None
    delegate_to: Optional[str] = None
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def tool_call(cls, name: str, args: Dict[str, Any]) -> Action:
        """Create a tool call action."""
        return cls(type=ActionType.TOOL_CALL, tool_name=name, tool_args=args)

    @classmethod
    def code_exec(cls, code: str) -> Action:
        """Create a code execution action."""
        return cls(type=ActionType.CODE_EXEC, code=code)

    @classmethod
    def respond(cls, response: str) -> Action:
        """Create a response action."""
        return cls(type=ActionType.RESPOND, response=response)

    @classmethod
    def create_tool(
        cls,
        name: str,
        description: str,
        code: str,
    ) -> Action:
        """Create a tool creation action."""
        return cls(
            type=ActionType.CREATE_TOOL,
            metadata={
                "tool_name": name,
                "tool_description": description,
                "tool_code": code,
            }
        )


@dataclass
class ActionResult:
    """Result of executing an action."""
    success: bool
    result: Any = None
    error: Optional[str] = None
    # For tool calls
    tool_output: Optional[str] = None
    # For code execution
    code_output: Optional[str] = None
    variables_created: List[str] = field(default_factory=list)
    # Execution time
    execution_time: float = 0.0
    # Whether task is complete
    task_complete: bool = False

    @classmethod
    def ok(cls, result: Any, **kwargs: Any) -> ActionResult:
        """Create a successful result."""
        return cls(success=True, result=result, **kwargs)

    @classmethod
    def fail(cls, error: str, **kwargs: Any) -> ActionResult:
        """Create a failed result."""
        return cls(success=False, error=error, **kwargs)


class BaseAgent(ABC):
    """
    Abstract base class for agents.

    Agents combine neural (LLM) and symbolic (code) computation
    to solve tasks.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Agent name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Agent description."""
        pass

    @abstractmethod
    async def think(self, input: str, context: Context) -> ThinkResult:
        """
        Think about the input and decide on an action.

        This is the neural computation step.

        Args:
            input: User input or task description
            context: Current context

        Returns:
            ThinkResult with reasoning and chosen action
        """
        pass

    @abstractmethod
    async def act(self, action: Action, context: Context) -> ActionResult:
        """
        Execute an action.

        This is the symbolic computation step.

        Args:
            action: Action to execute
            context: Current context

        Returns:
            ActionResult with the outcome
        """
        pass

    async def run(
        self,
        input: str,
        context: Optional[Context] = None,
        max_iterations: int = 10,
    ) -> str:
        """
        Run the agent on an input.

        Implements the think-act loop.

        Args:
            input: User input
            context: Optional initial context
            max_iterations: Maximum iterations

        Returns:
            Final response
        """
        if context is None:
            context = Context(user_input=input)

        iterations = 0
        final_response = ""

        while iterations < max_iterations:
            # Think
            think_result = await self.think(input, context)

            # Act
            action_result = await self.act(think_result.action, context)

            # Check if we should respond
            if think_result.action.type == ActionType.RESPOND:
                final_response = think_result.action.response or ""
                break

            # Check if task is complete
            if action_result.task_complete:
                final_response = str(action_result.result)
                break

            # Update context for next iteration
            # (subclasses should implement context updates)

            iterations += 1

        return final_response
