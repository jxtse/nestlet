"""
Tool Factory for dynamic tool creation.

Implements the LATM (LLMs as Tool Makers) paradigm:
1. Receive tool request from LLM
2. Generate tool code
3. Validate and test
4. Register as new tool
"""

from __future__ import annotations

import ast
import hashlib
import logging
import textwrap
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from inception.tool.base import (
    Tool,
    ToolSpec,
    ToolResult,
    ParameterSpec,
    ParameterType,
    ReturnSpec,
)
from inception.tool.registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of tool code validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    ast_node: Optional[ast.AST] = None


@dataclass
class TestResult:
    """Result of tool testing."""
    passed: bool
    output: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0


class CodeValidator:
    """
    Validates generated tool code for safety and correctness.
    """

    # Modules that should never be imported (network and dangerous modules only)
    BLOCKED_MODULES: Set[str] = {
        "socket", "requests", "urllib", "http", "ftplib", "smtplib",
        "telnetlib", "marshal", "ctypes",
    }

    # Dangerous built-in functions (allow open for file reading)
    BLOCKED_BUILTINS: Set[str] = {
        "eval", "exec", "compile", "__import__",
        "input", "breakpoint",
    }

    # Dangerous attributes
    BLOCKED_ATTRIBUTES: Set[str] = {
        "__class__", "__bases__", "__subclasses__",
        "__globals__", "__code__", "__builtins__",
    }

    def __init__(
        self,
        allowed_modules: Optional[Set[str]] = None,
        blocked_modules: Optional[Set[str]] = None,
    ):
        self.allowed_modules = allowed_modules or set()
        self.blocked_modules = blocked_modules or self.BLOCKED_MODULES

    def validate(self, code: str) -> ValidationResult:
        """
        Validate tool code.

        Args:
            code: Python source code

        Returns:
            ValidationResult with status and any issues
        """
        errors = []
        warnings = []

        # Parse the code
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Syntax error: {e}"],
                warnings=[],
            )

        # Check for blocked patterns
        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in self.blocked_modules:
                        errors.append(f"Blocked module import: {alias.name}")

            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] in self.blocked_modules:
                    errors.append(f"Blocked module import: {node.module}")

            # Check function calls
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in self.BLOCKED_BUILTINS:
                        errors.append(f"Blocked builtin function: {node.func.id}")

            # Check attribute access
            elif isinstance(node, ast.Attribute):
                if node.attr in self.BLOCKED_ATTRIBUTES:
                    errors.append(f"Blocked attribute access: {node.attr}")

        # Check for function definition
        func_defs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if not func_defs:
            errors.append("No function definition found in code")
        elif len(func_defs) > 1:
            warnings.append("Multiple function definitions found, using the first one")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            ast_node=tree,
        )


class GeneratedTool(Tool):
    """A tool created dynamically from generated code."""

    def __init__(
        self,
        spec: ToolSpec,
        func: Any,
    ):
        self._spec = spec
        self._func = func

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    async def execute(self, **kwargs: Any) -> ToolResult:
        import asyncio

        start_time = time.time()

        try:
            if asyncio.iscoroutinefunction(self._func):
                result = await self._func(**kwargs)
            else:
                result = self._func(**kwargs)

            return ToolResult.ok(
                result=result,
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            return ToolResult.fail(
                error=str(e),
                execution_time=time.time() - start_time,
            )


class ToolFactory:
    """
    Factory for creating tools dynamically.

    The LATM workflow:
    1. create_from_code: Generate tool from code string
    2. Validate code safety
    3. Create executable function
    4. Test with sample inputs
    5. Register if successful
    """

    def __init__(
        self,
        registry: ToolRegistry,
        validator: Optional[CodeValidator] = None,
        sandbox_globals: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the factory.

        Args:
            registry: Tool registry to register new tools
            validator: Code validator (creates default if not provided)
            sandbox_globals: Global namespace for execution
        """
        self.registry = registry
        self.validator = validator or CodeValidator()
        self._sandbox_globals = sandbox_globals or self._create_safe_globals()

    def _create_safe_globals(self) -> Dict[str, Any]:
        """Create a safe global namespace for tool execution."""
        import math
        import statistics
        import json
        import re
        import datetime
        import os
        import sys
        import pathlib
        import io
        import csv
        import glob as glob_module
        from collections import Counter, defaultdict
        from itertools import chain, combinations, permutations
        from functools import reduce

        safe_globals = {
            "__builtins__": {
                # Safe builtins
                "abs": abs, "all": all, "any": any,
                "bin": bin, "bool": bool, "bytes": bytes,
                "chr": chr, "dict": dict, "divmod": divmod,
                "enumerate": enumerate, "filter": filter,
                "float": float, "format": format, "frozenset": frozenset,
                "hash": hash, "hex": hex, "int": int, "isinstance": isinstance,
                "issubclass": issubclass, "iter": iter, "len": len,
                "list": list, "map": map, "max": max, "min": min,
                "next": next, "oct": oct, "ord": ord, "pow": pow,
                "print": print, "range": range, "repr": repr,
                "reversed": reversed, "round": round, "set": set,
                "slice": slice, "sorted": sorted, "str": str,
                "sum": sum, "tuple": tuple, "type": type, "zip": zip,
                # File operations (needed for data analysis)
                "open": open,
                # Exceptions
                "Exception": Exception, "ValueError": ValueError,
                "TypeError": TypeError, "KeyError": KeyError,
                "IndexError": IndexError, "RuntimeError": RuntimeError,
                "FileNotFoundError": FileNotFoundError, "IOError": IOError,
            },
            # Safe modules
            "math": math,
            "statistics": statistics,
            "json": json,
            "re": re,
            "datetime": datetime,
            "Counter": Counter,
            "defaultdict": defaultdict,
            "chain": chain,
            "combinations": combinations,
            "permutations": permutations,
            "reduce": reduce,
            # File system modules
            "os": os,
            "sys": sys,
            "pathlib": pathlib,
            "Path": pathlib.Path,
            "io": io,
            "csv": csv,
            "glob": glob_module,
        }

        # Try to add numpy/pandas if available
        try:
            import numpy as np
            safe_globals["np"] = np
            safe_globals["numpy"] = np
        except ImportError:
            pass

        try:
            import pandas as pd
            safe_globals["pd"] = pd
            safe_globals["pandas"] = pd
        except ImportError:
            pass

        return safe_globals

    def create_from_code(
        self,
        name: str,
        description: str,
        code: str,
        parameters: Optional[Dict[str, ParameterSpec]] = None,
        returns: Optional[ReturnSpec] = None,
        category: str = "generated",
        tags: Optional[List[str]] = None,
        test_cases: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Tool]:
        """
        Create a tool from Python code.

        Args:
            name: Tool name
            description: Tool description
            code: Python code containing the tool function
            parameters: Parameter specifications (auto-detected if not provided)
            returns: Return specification
            category: Tool category
            tags: Tool tags
            test_cases: Optional test cases to validate the tool

        Returns:
            Created tool or None if validation/testing failed
        """
        # Step 1: Validate code
        validation = self.validator.validate(code)
        if not validation.is_valid:
            logger.error(f"Code validation failed: {validation.errors}")
            return None

        for warning in validation.warnings:
            logger.warning(f"Code validation warning: {warning}")

        # Step 2: Execute code to get function
        try:
            exec_globals = dict(self._sandbox_globals)
            exec(code, exec_globals)

            # Find the function
            func = None
            for val in exec_globals.values():
                if callable(val) and not isinstance(val, type):
                    func = val
                    break

            if func is None:
                logger.error("No callable function found in code")
                return None

        except Exception as e:
            logger.error(f"Code execution failed: {e}")
            return None

        # Step 3: Auto-detect parameters if not provided
        if parameters is None:
            parameters = self._detect_parameters(func)

        # Step 4: Create spec
        spec = ToolSpec(
            name=name,
            description=description,
            parameters=parameters,
            returns=returns,
            category=category,
            tags=tags or [],
            source_code=code,
            is_generated=True,
        )

        # Step 5: Create tool
        tool = GeneratedTool(spec, func)

        # Step 6: Run test cases if provided
        if test_cases:
            for i, test_case in enumerate(test_cases):
                result = self._run_test(tool, test_case)
                if not result.passed:
                    logger.error(f"Test case {i+1} failed: {result.error}")
                    return None

        logger.info(f"Successfully created tool: {name}")
        return tool

    def _detect_parameters(self, func: Any) -> Dict[str, ParameterSpec]:
        """Auto-detect parameters from function signature."""
        import inspect
        from typing import get_type_hints

        parameters = {}
        sig = inspect.signature(func)
        hints = {}

        try:
            hints = get_type_hints(func)
        except Exception:
            pass

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

            param_type = hints.get(param_name)
            ptype = type_mapping.get(param_type, ParameterType.STRING)
            has_default = param.default != inspect.Parameter.empty

            parameters[param_name] = ParameterSpec(
                name=param_name,
                type=ptype,
                description=f"Parameter: {param_name}",
                required=not has_default,
                default=param.default if has_default else None,
            )

        return parameters

    def _run_test(self, tool: Tool, test_case: Dict[str, Any]) -> TestResult:
        """Run a single test case."""
        import asyncio

        inputs = test_case.get("inputs", {})
        expected = test_case.get("expected")

        try:
            # Run synchronously for testing
            loop = asyncio.new_event_loop()
            result = loop.run_until_complete(tool.execute(**inputs))
            loop.close()

            if not result.success:
                return TestResult(passed=False, error=result.error)

            if expected is not None and result.result != expected:
                return TestResult(
                    passed=False,
                    output=result.result,
                    error=f"Expected {expected}, got {result.result}",
                )

            return TestResult(
                passed=True,
                output=result.result,
                execution_time=result.execution_time,
            )

        except Exception as e:
            return TestResult(passed=False, error=str(e))

    def create_and_register(
        self,
        name: str,
        description: str,
        code: str,
        **kwargs: Any,
    ) -> bool:
        """
        Create a tool and register it.

        Returns:
            True if successful
        """
        tool = self.create_from_code(name, description, code, **kwargs)
        if tool is None:
            return False

        self.registry.register(tool, is_builtin=False, override=True)
        return True

    def generate_tool_code_prompt(
        self,
        task_description: str,
        suggested_name: str,
    ) -> str:
        """
        Generate a prompt for the LLM to create tool code.

        Args:
            task_description: What the tool should do
            suggested_name: Suggested function name

        Returns:
            Prompt string for LLM
        """
        return textwrap.dedent(f"""
        Create a Python function to accomplish the following task:

        Task: {task_description}

        Requirements:
        1. Function name: {suggested_name}
        2. Include type hints for all parameters and return value
        3. Include a docstring describing what the function does
        4. Handle edge cases and errors gracefully
        5. Return a meaningful result (not just print)
        6. Do NOT use any external API calls or file operations
        7. Use only these safe modules: math, statistics, json, re, datetime,
           collections (Counter, defaultdict), itertools, functools, numpy, pandas

        Example format:
        ```python
        def {suggested_name}(param1: type1, param2: type2 = default) -> return_type:
            '''
            Description of what the function does.

            Args:
                param1: Description
                param2: Description

            Returns:
                Description of return value
            '''
            # Implementation
            return result
        ```

        Provide only the Python code, no explanations.
        """).strip()
