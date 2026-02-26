"""
File Operation Tools.

Provides safe file operations within a workspace.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import aiofiles

from inception.tool.base import (
    Tool,
    ToolSpec,
    ToolResult,
    ParameterSpec,
    ParameterType,
    ReturnSpec,
)


class ReadFileTool(Tool):
    """Tool for reading files from the workspace."""

    def __init__(self, workspace: Path):
        """
        Initialize with workspace path.

        Args:
            workspace: Root directory for file operations
        """
        self._workspace = Path(workspace).resolve()
        self._spec = ToolSpec(
            name="read_file",
            description=(
                "Read the contents of a file from the workspace. "
                "Only files within the workspace directory can be accessed."
            ),
            parameters={
                "path": ParameterSpec(
                    name="path",
                    type=ParameterType.STRING,
                    description="Relative path to the file within the workspace",
                    required=True,
                ),
                "encoding": ParameterSpec(
                    name="encoding",
                    type=ParameterType.STRING,
                    description="File encoding (default: utf-8)",
                    required=False,
                    default="utf-8",
                ),
            },
            returns=ReturnSpec(
                type=ParameterType.STRING,
                description="Contents of the file",
            ),
            category="file",
            tags=["file", "read", "io"],
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def _validate_path(self, path: str) -> Optional[Path]:
        """Validate and resolve path within workspace."""
        try:
            # Resolve the full path
            full_path = (self._workspace / path).resolve()

            # Ensure it's within the workspace
            if not str(full_path).startswith(str(self._workspace)):
                return None

            return full_path
        except Exception:
            return None

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Read a file."""
        path = kwargs.get("path", "")
        encoding = kwargs.get("encoding", "utf-8")

        if not path:
            return ToolResult.fail("No path provided")

        full_path = self._validate_path(path)
        if full_path is None:
            return ToolResult.fail("Invalid path: must be within workspace")

        if not full_path.exists():
            return ToolResult.fail(f"File not found: {path}")

        if not full_path.is_file():
            return ToolResult.fail(f"Not a file: {path}")

        try:
            async with aiofiles.open(full_path, "r", encoding=encoding) as f:
                content = await f.read()
            return ToolResult.ok(result=content)
        except Exception as e:
            return ToolResult.fail(f"Failed to read file: {e}")


class WriteFileTool(Tool):
    """Tool for writing files to the workspace."""

    def __init__(self, workspace: Path):
        """
        Initialize with workspace path.

        Args:
            workspace: Root directory for file operations
        """
        self._workspace = Path(workspace).resolve()
        self._spec = ToolSpec(
            name="write_file",
            description=(
                "Write content to a file in the workspace. "
                "Creates parent directories if needed. "
                "Only files within the workspace directory can be written."
            ),
            parameters={
                "path": ParameterSpec(
                    name="path",
                    type=ParameterType.STRING,
                    description="Relative path to the file within the workspace",
                    required=True,
                ),
                "content": ParameterSpec(
                    name="content",
                    type=ParameterType.STRING,
                    description="Content to write to the file",
                    required=True,
                ),
                "encoding": ParameterSpec(
                    name="encoding",
                    type=ParameterType.STRING,
                    description="File encoding (default: utf-8)",
                    required=False,
                    default="utf-8",
                ),
                "append": ParameterSpec(
                    name="append",
                    type=ParameterType.BOOLEAN,
                    description="Append to file instead of overwriting",
                    required=False,
                    default=False,
                ),
            },
            returns=ReturnSpec(
                type=ParameterType.OBJECT,
                description="Result with path and bytes written",
            ),
            category="file",
            tags=["file", "write", "io"],
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def _validate_path(self, path: str) -> Optional[Path]:
        """Validate and resolve path within workspace."""
        try:
            full_path = (self._workspace / path).resolve()
            if not str(full_path).startswith(str(self._workspace)):
                return None
            return full_path
        except Exception:
            return None

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Write a file."""
        path = kwargs.get("path", "")
        content = kwargs.get("content", "")
        encoding = kwargs.get("encoding", "utf-8")
        append = kwargs.get("append", False)

        if not path:
            return ToolResult.fail("No path provided")

        full_path = self._validate_path(path)
        if full_path is None:
            return ToolResult.fail("Invalid path: must be within workspace")

        try:
            # Create parent directories
            full_path.parent.mkdir(parents=True, exist_ok=True)

            mode = "a" if append else "w"
            async with aiofiles.open(full_path, mode, encoding=encoding) as f:
                await f.write(content)

            return ToolResult.ok(result={
                "path": str(full_path.relative_to(self._workspace)),
                "bytes_written": len(content.encode(encoding)),
                "mode": "appended" if append else "written",
            })
        except Exception as e:
            return ToolResult.fail(f"Failed to write file: {e}")


class ListDirectoryTool(Tool):
    """Tool for listing directory contents."""

    def __init__(self, workspace: Path):
        """
        Initialize with workspace path.

        Args:
            workspace: Root directory for file operations
        """
        self._workspace = Path(workspace).resolve()
        self._spec = ToolSpec(
            name="list_directory",
            description=(
                "List contents of a directory in the workspace. "
                "Returns files and subdirectories with metadata."
            ),
            parameters={
                "path": ParameterSpec(
                    name="path",
                    type=ParameterType.STRING,
                    description="Relative path to the directory (default: workspace root)",
                    required=False,
                    default=".",
                ),
                "recursive": ParameterSpec(
                    name="recursive",
                    type=ParameterType.BOOLEAN,
                    description="List recursively",
                    required=False,
                    default=False,
                ),
                "pattern": ParameterSpec(
                    name="pattern",
                    type=ParameterType.STRING,
                    description="Glob pattern to filter files (e.g., '*.py')",
                    required=False,
                ),
            },
            returns=ReturnSpec(
                type=ParameterType.ARRAY,
                description="List of file/directory entries",
            ),
            category="file",
            tags=["file", "list", "directory"],
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def _validate_path(self, path: str) -> Optional[Path]:
        """Validate and resolve path within workspace."""
        try:
            full_path = (self._workspace / path).resolve()
            if not str(full_path).startswith(str(self._workspace)):
                return None
            return full_path
        except Exception:
            return None

    async def execute(self, **kwargs: Any) -> ToolResult:
        """List directory contents."""
        path = kwargs.get("path", ".")
        recursive = kwargs.get("recursive", False)
        pattern = kwargs.get("pattern")

        full_path = self._validate_path(path)
        if full_path is None:
            return ToolResult.fail("Invalid path: must be within workspace")

        if not full_path.exists():
            return ToolResult.fail(f"Directory not found: {path}")

        if not full_path.is_dir():
            return ToolResult.fail(f"Not a directory: {path}")

        try:
            entries = []

            if recursive:
                if pattern:
                    items = full_path.rglob(pattern)
                else:
                    items = full_path.rglob("*")
            else:
                if pattern:
                    items = full_path.glob(pattern)
                else:
                    items = full_path.iterdir()

            for item in items:
                rel_path = item.relative_to(self._workspace)
                entries.append({
                    "name": item.name,
                    "path": str(rel_path),
                    "is_file": item.is_file(),
                    "is_dir": item.is_dir(),
                    "size": item.stat().st_size if item.is_file() else None,
                })

            return ToolResult.ok(result=entries)
        except Exception as e:
            return ToolResult.fail(f"Failed to list directory: {e}")
