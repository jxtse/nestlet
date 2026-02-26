"""
Tool base classes and specifications.

Defines the interface for all tools in the Inception system.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union

from inception.provider.base import ToolDefinition


class ParameterType(str, Enum):
    """Supported parameter types."""
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


@dataclass
class ParameterSpec:
    """Specification for a tool parameter."""
    name: str
    type: ParameterType
    description: str
    required: bool = True
    default: Optional[Any] = None
    enum: Optional[List[Any]] = None
    # For array type
    items_type: Optional[ParameterType] = None
    # For object type
    properties: Optional[Dict[str, "ParameterSpec"]] = None

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema format."""
        schema: Dict[str, Any] = {
            "type": self.type.value,
            "description": self.description,
        }
        if self.enum:
            schema["enum"] = self.enum
        if self.default is not None:
            schema["default"] = self.default
        if self.type == ParameterType.ARRAY and self.items_type:
            schema["items"] = {"type": self.items_type.value}
        if self.type == ParameterType.OBJECT and self.properties:
            schema["properties"] = {
                name: spec.to_json_schema()
                for name, spec in self.properties.items()
            }
        return schema


@dataclass
class ReturnSpec:
    """Specification for tool return value."""
    type: ParameterType
    description: str
    # For structured returns
    properties: Optional[Dict[str, ParameterSpec]] = None

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema format."""
        schema: Dict[str, Any] = {
            "type": self.type.value,
            "description": self.description,
        }
        if self.properties:
            schema["properties"] = {
                name: spec.to_json_schema()
                for name, spec in self.properties.items()
            }
        return schema


@dataclass
class ToolSpec:
    """
    Complete specification for a tool.

    Used for:
    - LLM tool calling (converted to JSON Schema)
    - Tool documentation
    - Validation
    """
    name: str
    description: str
    parameters: Dict[str, ParameterSpec] = field(default_factory=dict)
    returns: Optional[ReturnSpec] = None
    # Metadata
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    # Source info for generated tools
    source_code: Optional[str] = None
    is_generated: bool = False

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert parameters to JSON Schema for LLM tool calling."""
        required = [
            name for name, spec in self.parameters.items()
            if spec.required
        ]

        return {
            "type": "object",
            "properties": {
                name: spec.to_json_schema()
                for name, spec in self.parameters.items()
            },
            "required": required,
        }

    def to_tool_definition(self) -> ToolDefinition:
        """Convert to ToolDefinition for provider."""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.to_json_schema(),
        )


@dataclass
class ToolResult:
    """Result from tool execution."""
    success: bool
    result: Any = None
    error: Optional[str] = None
    # Execution metadata
    execution_time: float = 0.0
    # For stateful execution
    variables_created: List[str] = field(default_factory=list)
    variables_modified: List[str] = field(default_factory=list)

    @classmethod
    def ok(cls, result: Any, **kwargs: Any) -> ToolResult:
        """Create a successful result."""
        return cls(success=True, result=result, **kwargs)

    @classmethod
    def fail(cls, error: str, **kwargs: Any) -> ToolResult:
        """Create a failed result."""
        return cls(success=False, error=error, **kwargs)


class Tool(ABC):
    """
    Abstract base class for tools.

    Tools are the primary mechanism for symbolic computation.
    They can be:
    - Built-in (predefined in code)
    - Generated (created by LLM at runtime)
    """

    @property
    @abstractmethod
    def spec(self) -> ToolSpec:
        """Get the tool specification."""
        pass

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """
        Execute the tool with the given arguments.

        Args:
            **kwargs: Tool-specific arguments

        Returns:
            ToolResult with the outcome
        """
        pass

    def validate_args(self, **kwargs: Any) -> Optional[str]:
        """
        Validate arguments against the spec.

        Returns:
            Error message if validation fails, None otherwise
        """
        for name, param_spec in self.spec.parameters.items():
            if param_spec.required and name not in kwargs:
                return f"Missing required parameter: {name}"

            if name in kwargs:
                value = kwargs[name]
                # Type validation (basic)
                if param_spec.type == ParameterType.STRING and not isinstance(value, str):
                    return f"Parameter {name} must be a string"
                elif param_spec.type == ParameterType.INTEGER and not isinstance(value, int):
                    return f"Parameter {name} must be an integer"
                elif param_spec.type == ParameterType.NUMBER and not isinstance(value, (int, float)):
                    return f"Parameter {name} must be a number"
                elif param_spec.type == ParameterType.BOOLEAN and not isinstance(value, bool):
                    return f"Parameter {name} must be a boolean"
                elif param_spec.type == ParameterType.ARRAY and not isinstance(value, list):
                    return f"Parameter {name} must be an array"
                elif param_spec.type == ParameterType.OBJECT and not isinstance(value, dict):
                    return f"Parameter {name} must be an object"

                # Enum validation
                if param_spec.enum and value not in param_spec.enum:
                    return f"Parameter {name} must be one of: {param_spec.enum}"

        return None

    async def __call__(self, **kwargs: Any) -> ToolResult:
        """Convenience method to call execute."""
        # Validate arguments
        error = self.validate_args(**kwargs)
        if error:
            return ToolResult.fail(error)

        return await self.execute(**kwargs)


class FunctionTool(Tool):
    """
    A tool wrapper for simple functions.

    Allows wrapping any async function as a tool.
    """

    def __init__(
        self,
        func: Any,
        spec: ToolSpec,
    ):
        """
        Create a function-based tool.

        Args:
            func: The async function to wrap
            spec: Tool specification
        """
        self._func = func
        self._spec = spec

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    async def execute(self, **kwargs: Any) -> ToolResult:
        import asyncio
        import time

        start_time = time.time()

        try:
            # Check if function is async
            if asyncio.iscoroutinefunction(self._func):
                result = await self._func(**kwargs)
            else:
                result = self._func(**kwargs)

            return ToolResult.ok(
                result=result,
                execution_time=time.time() - start_time
            )
        except Exception as e:
            return ToolResult.fail(
                error=str(e),
                execution_time=time.time() - start_time
            )


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    category: str = "general",
    **kwargs: Any
):
    """
    Decorator to create a tool from a function.

    Usage:
        @tool(name="add_numbers", description="Add two numbers")
        async def add(a: int, b: int) -> int:
            return a + b
    """
    def decorator(func):
        import inspect
        from typing import get_type_hints

        # Get function metadata
        func_name = name or func.__name__
        func_description = description or func.__doc__ or f"Tool: {func_name}"

        # Parse parameters from type hints
        hints = get_type_hints(func) if hasattr(func, "__annotations__") else {}
        sig = inspect.signature(func)
        params = {}

        type_mapping = {
            str: ParameterType.STRING,
            int: ParameterType.INTEGER,
            float: ParameterType.NUMBER,
            bool: ParameterType.BOOLEAN,
            list: ParameterType.ARRAY,
            dict: ParameterType.OBJECT,
        }

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            param_type = hints.get(param_name, Any)
            ptype = type_mapping.get(param_type, ParameterType.STRING)

            has_default = param.default != inspect.Parameter.empty

            params[param_name] = ParameterSpec(
                name=param_name,
                type=ptype,
                description=f"Parameter: {param_name}",
                required=not has_default,
                default=param.default if has_default else None,
            )

        # Create spec
        spec = ToolSpec(
            name=func_name,
            description=func_description,
            parameters=params,
            category=category,
        )

        return FunctionTool(func, spec)

    return decorator
