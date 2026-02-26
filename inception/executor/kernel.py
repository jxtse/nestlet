"""
Python Kernel for stateful code execution.

Implements an IPython-based kernel that maintains state across executions.
Based on TaskWeaver's kernel design.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import traceback
from dataclasses import dataclass, field
from io import StringIO
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result from code execution."""
    success: bool
    output: str = ""
    result: Any = None
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None
    error_type: Optional[str] = None
    traceback: Optional[str] = None
    # Execution metadata
    variables_created: List[str] = field(default_factory=list)
    variables_modified: List[str] = field(default_factory=list)
    execution_count: int = 0


class PythonKernel:
    """
    Stateful Python execution kernel.

    Features:
    - Variables persist across executions
    - Pre-loaded common libraries
    - Captures stdout/stderr
    - Safe execution with timeout support
    """

    # Default modules to pre-import
    DEFAULT_IMPORTS = """
import math
import statistics
import json
import re
import datetime
import os
import sys
import pathlib
from pathlib import Path
import io
import csv
import glob
from collections import Counter, defaultdict, OrderedDict
from typing import List, Dict, Any, Optional, Tuple, Set
from functools import reduce, partial
from itertools import chain, combinations, permutations, product
"""

    OPTIONAL_IMPORTS = """
try:
    import numpy as np
except ImportError:
    pass

try:
    import pandas as pd
except ImportError:
    pass

try:
    import openpyxl
except ImportError:
    pass

try:
    import xlrd
except ImportError:
    pass

try:
    import olefile
except ImportError:
    pass
"""

    def __init__(
        self,
        allowed_modules: Optional[Set[str]] = None,
        blocked_modules: Optional[Set[str]] = None,
        timeout: float = 30.0,
    ):
        """
        Initialize the kernel.

        Args:
            allowed_modules: Whitelist of allowed modules (None for default)
            blocked_modules: Blacklist of blocked modules
            timeout: Default execution timeout in seconds
        """
        self._namespace: Dict[str, Any] = {}
        self._execution_count = 0
        self._timeout = timeout
        self._allowed_modules = allowed_modules
        # Default blocked modules - only block network and truly dangerous modules
        # Allow os, sys, pathlib for file operations needed by data analysis
        self._blocked_modules = blocked_modules or {
            "socket", "requests", "urllib", "http", "ftplib",
            "smtplib", "telnetlib", "asyncio.subprocess",
        }
        self._initialized = False
        self._history: List[str] = []

    async def initialize(self) -> None:
        """Initialize the kernel with default imports."""
        if self._initialized:
            return

        # Set up safe builtins
        self._namespace["__builtins__"] = self._create_safe_builtins()

        # Run default imports
        await self.execute(self.DEFAULT_IMPORTS, capture_result=False)
        await self.execute(self.OPTIONAL_IMPORTS, capture_result=False)

        self._initialized = True
        logger.info("Python kernel initialized")

    def _create_safe_builtins(self) -> Dict[str, Any]:
        """Create a restricted set of builtins."""
        import builtins

        # Start with all builtins
        safe_builtins = dict(vars(builtins))

        # Remove dangerous functions
        # Note: open is allowed for file reading (needed for data analysis)
        # eval/exec/compile are restricted to prevent code injection
        dangerous = [
            "eval", "exec", "compile", "__import__",
            "input", "breakpoint",
        ]
        for name in dangerous:
            safe_builtins.pop(name, None)

        # Add back a restricted __import__
        safe_builtins["__import__"] = self._restricted_import

        return safe_builtins

    def _restricted_import(
        self,
        name: str,
        globals: Optional[Dict] = None,
        locals: Optional[Dict] = None,
        fromlist: tuple = (),
        level: int = 0,
    ) -> Any:
        """Restricted import function that checks against blocked modules."""
        # Get the top-level module name
        top_level = name.split(".")[0]

        if top_level in self._blocked_modules:
            raise ImportError(f"Module '{name}' is not allowed")

        if self._allowed_modules and top_level not in self._allowed_modules:
            raise ImportError(f"Module '{name}' is not in the allowed list")

        # Use the real __import__
        return __builtins__["__import__"](name, globals, locals, fromlist, level)

    async def execute(
        self,
        code: str,
        timeout: Optional[float] = None,
        capture_result: bool = True,
    ) -> ExecutionResult:
        """
        Execute Python code.

        Args:
            code: Python code to execute
            timeout: Execution timeout (uses default if not specified)
            capture_result: Whether to capture the result of the last expression

        Returns:
            ExecutionResult with outputs and any errors
        """
        if not self._initialized and "import" not in code[:50]:
            await self.initialize()

        timeout = timeout or self._timeout
        self._execution_count += 1

        # Track variables before execution
        vars_before = set(self._namespace.keys())

        # Capture stdout/stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        stdout_capture = StringIO()
        stderr_capture = StringIO()

        result = ExecutionResult(
            success=False,
            execution_count=self._execution_count,
        )

        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            # Run in executor to support timeout
            loop = asyncio.get_event_loop()
            await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self._exec_code_sync(code, capture_result, result),
                ),
                timeout=timeout,
            )

        except asyncio.TimeoutError:
            result.success = False
            result.error = f"Execution timed out after {timeout} seconds"
            result.error_type = "TimeoutError"

        except Exception as e:
            result.success = False
            result.error = str(e)
            result.error_type = type(e).__name__
            result.traceback = traceback.format_exc()

        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            result.stdout = stdout_capture.getvalue()
            result.stderr = stderr_capture.getvalue()

        # Track variable changes
        vars_after = set(self._namespace.keys())
        result.variables_created = list(vars_after - vars_before)
        result.variables_modified = []  # Would need more sophisticated tracking

        # Build output string
        if result.success:
            outputs = []
            if result.stdout:
                outputs.append(result.stdout)
            if result.result is not None:
                outputs.append(repr(result.result))
            result.output = "\n".join(outputs)
        else:
            result.output = result.error or ""

        # Add to history
        self._history.append(code)

        return result

    def _exec_code_sync(
        self,
        code: str,
        capture_result: bool,
        result: ExecutionResult,
    ) -> None:
        """Execute code synchronously."""
        try:
            # Try to compile as expression first (to capture result)
            if capture_result:
                try:
                    # Try to get last expression
                    import ast
                    tree = ast.parse(code)

                    if tree.body and isinstance(tree.body[-1], ast.Expr):
                        # Execute all but last statement
                        if len(tree.body) > 1:
                            exec_code = ast.Module(body=tree.body[:-1], type_ignores=[])
                            exec(compile(exec_code, "<kernel>", "exec"), self._namespace)

                        # Evaluate last expression
                        last_expr = ast.Expression(body=tree.body[-1].value)
                        result.result = eval(
                            compile(last_expr, "<kernel>", "eval"),
                            self._namespace,
                        )
                    else:
                        exec(code, self._namespace)

                except SyntaxError:
                    # Fall back to simple exec
                    exec(code, self._namespace)
            else:
                exec(code, self._namespace)

            result.success = True

        except Exception as e:
            result.success = False
            result.error = str(e)
            result.error_type = type(e).__name__
            result.traceback = traceback.format_exc()

    def get_variable(self, name: str) -> Any:
        """Get a variable from the namespace."""
        return self._namespace.get(name)

    def set_variable(self, name: str, value: Any) -> None:
        """Set a variable in the namespace."""
        self._namespace[name] = value

    def delete_variable(self, name: str) -> bool:
        """Delete a variable from the namespace."""
        if name in self._namespace:
            del self._namespace[name]
            return True
        return False

    def list_variables(self) -> List[str]:
        """List all user-defined variables."""
        # Filter out modules, builtins, and special names
        return [
            name for name in self._namespace.keys()
            if not name.startswith("_")
            and not isinstance(self._namespace[name], type(sys))
        ]

    def get_variable_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a variable."""
        if name not in self._namespace:
            return None

        value = self._namespace[name]
        info = {
            "name": name,
            "type": type(value).__name__,
        }

        # Add size info for collections
        if hasattr(value, "__len__"):
            try:
                info["length"] = len(value)
            except Exception:
                pass

        # Add shape for numpy arrays / pandas DataFrames
        if hasattr(value, "shape"):
            try:
                info["shape"] = value.shape
            except Exception:
                pass

        # Add preview for small values
        try:
            repr_str = repr(value)
            if len(repr_str) < 100:
                info["preview"] = repr_str
            else:
                info["preview"] = repr_str[:97] + "..."
        except Exception:
            pass

        return info

    def reset(self) -> None:
        """Reset the kernel to initial state."""
        self._namespace.clear()
        self._execution_count = 0
        self._initialized = False
        self._history.clear()
        logger.info("Python kernel reset")

    @property
    def execution_count(self) -> int:
        """Get the current execution count."""
        return self._execution_count

    @property
    def history(self) -> List[str]:
        """Get execution history."""
        return self._history.copy()
