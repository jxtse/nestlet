"""
Code Execution Tool.

Provides the ability to execute Python code in a managed kernel.
"""

from __future__ import annotations

import time
from typing import Any, TYPE_CHECKING

from inception.tool.base import (
    Tool,
    ToolSpec,
    ToolResult,
    ParameterSpec,
    ParameterType,
    ReturnSpec,
)

if TYPE_CHECKING:
    from inception.executor.kernel import PythonKernel


class CodeExecutionTool(Tool):
    """
    Tool for executing Python code in a stateful kernel.

    Features:
    - Variables persist across executions
    - Access to common data science libraries
    - Captures stdout, stderr, and return values
    """

    def __init__(self, kernel: "PythonKernel"):
        """
        Initialize with a Python kernel.

        Args:
            kernel: The kernel for code execution
        """
        self._kernel = kernel
        self._spec = ToolSpec(
            name="execute_code",
            description=(
                "Execute Python code in a stateful environment. "
                "Variables and imports persist across calls. "
                "Available libraries: numpy, pandas, math, statistics, json, re, datetime. "
                "Returns the result of the last expression or explicit return value."
            ),
            parameters={
                "code": ParameterSpec(
                    name="code",
                    type=ParameterType.STRING,
                    description="Python code to execute",
                    required=True,
                ),
            },
            returns=ReturnSpec(
                type=ParameterType.OBJECT,
                description="Execution result with output and any errors",
            ),
            category="execution",
            tags=["python", "code", "computation"],
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute Python code."""
        code = kwargs.get("code", "")

        if not code.strip():
            return ToolResult.fail("No code provided")

        start_time = time.time()

        try:
            result = await self._kernel.execute(code)

            # Track variable changes
            variables_created = []
            variables_modified = []

            if result.success:
                return ToolResult.ok(
                    result={
                        "output": result.output,
                        "result": result.result,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                    },
                    execution_time=time.time() - start_time,
                    variables_created=variables_created,
                    variables_modified=variables_modified,
                )
            else:
                return ToolResult.fail(
                    error=result.error or "Unknown error",
                    execution_time=time.time() - start_time,
                )

        except Exception as e:
            return ToolResult.fail(
                error=str(e),
                execution_time=time.time() - start_time,
            )


class CodeAnalysisTool(Tool):
    """
    Tool for analyzing code without executing it.

    Useful for:
    - Understanding code structure
    - Extracting function signatures
    - Checking for syntax errors
    """

    def __init__(self):
        self._spec = ToolSpec(
            name="analyze_code",
            description=(
                "Analyze Python code structure without executing. "
                "Returns information about functions, classes, imports, and any syntax errors."
            ),
            parameters={
                "code": ParameterSpec(
                    name="code",
                    type=ParameterType.STRING,
                    description="Python code to analyze",
                    required=True,
                ),
            },
            returns=ReturnSpec(
                type=ParameterType.OBJECT,
                description="Analysis result with code structure information",
            ),
            category="execution",
            tags=["python", "code", "analysis"],
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Analyze Python code."""
        import ast

        code = kwargs.get("code", "")

        if not code.strip():
            return ToolResult.fail("No code provided")

        try:
            tree = ast.parse(code)

            # Extract information
            functions = []
            classes = []
            imports = []

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append({
                        "name": node.name,
                        "args": [arg.arg for arg in node.args.args],
                        "lineno": node.lineno,
                    })
                elif isinstance(node, ast.AsyncFunctionDef):
                    functions.append({
                        "name": node.name,
                        "args": [arg.arg for arg in node.args.args],
                        "lineno": node.lineno,
                        "async": True,
                    })
                elif isinstance(node, ast.ClassDef):
                    classes.append({
                        "name": node.name,
                        "lineno": node.lineno,
                    })
                elif isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)

            return ToolResult.ok(result={
                "valid": True,
                "functions": functions,
                "classes": classes,
                "imports": imports,
                "line_count": len(code.splitlines()),
            })

        except SyntaxError as e:
            return ToolResult.ok(result={
                "valid": False,
                "error": str(e),
                "line": e.lineno,
                "offset": e.offset,
            })
