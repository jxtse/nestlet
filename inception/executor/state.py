"""
State Management for the execution engine.

Tracks variables, data artifacts, and execution context.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class Variable:
    """Represents a variable in the execution state."""
    name: str
    type_name: str
    value_preview: str
    created_at: datetime
    modified_at: datetime
    # For collections
    length: Optional[int] = None
    # For arrays/dataframes
    shape: Optional[tuple] = None
    # For dataframes
    columns: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.type_name,
            "preview": self.value_preview,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
            "length": self.length,
            "shape": self.shape,
            "columns": self.columns,
        }


@dataclass
class DataArtifact:
    """Represents a data artifact (file, plot, etc.)."""
    name: str
    artifact_type: str  # "file", "plot", "dataframe", etc.
    path: Optional[Path] = None
    data: Optional[Any] = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionContext:
    """Context for a single execution step."""
    code: str
    result: Any
    success: bool
    timestamp: datetime
    variables_created: List[str] = field(default_factory=list)
    variables_modified: List[str] = field(default_factory=list)
    artifacts_created: List[str] = field(default_factory=list)
    error: Optional[str] = None


class StateManager:
    """
    Manages the execution state across multiple code executions.

    Features:
    - Track variables and their evolution
    - Store data artifacts
    - Provide context for the agent
    - Support state checkpointing
    """

    def __init__(self, workspace: Optional[Path] = None):
        """
        Initialize the state manager.

        Args:
            workspace: Optional workspace directory for artifacts
        """
        self._variables: Dict[str, Variable] = {}
        self._artifacts: Dict[str, DataArtifact] = {}
        self._history: List[ExecutionContext] = []
        self._workspace = workspace

        if workspace:
            workspace.mkdir(parents=True, exist_ok=True)

    def update_variable(
        self,
        name: str,
        value: Any,
        is_new: bool = False,
    ) -> None:
        """
        Update or create a variable entry.

        Args:
            name: Variable name
            value: Variable value
            is_new: Whether this is a new variable
        """
        now = datetime.now()

        # Get type information
        type_name = type(value).__name__

        # Get preview
        try:
            preview = repr(value)
            if len(preview) > 100:
                preview = preview[:97] + "..."
        except Exception:
            preview = f"<{type_name}>"

        # Get size information
        length = None
        shape = None
        columns = None

        if hasattr(value, "__len__"):
            try:
                length = len(value)
            except Exception:
                pass

        if hasattr(value, "shape"):
            try:
                shape = tuple(value.shape)
            except Exception:
                pass

        if hasattr(value, "columns"):
            try:
                columns = list(value.columns)
            except Exception:
                pass

        if name in self._variables and not is_new:
            # Update existing
            var = self._variables[name]
            var.type_name = type_name
            var.value_preview = preview
            var.modified_at = now
            var.length = length
            var.shape = shape
            var.columns = columns
        else:
            # Create new
            self._variables[name] = Variable(
                name=name,
                type_name=type_name,
                value_preview=preview,
                created_at=now,
                modified_at=now,
                length=length,
                shape=shape,
                columns=columns,
            )

    def remove_variable(self, name: str) -> bool:
        """Remove a variable from tracking."""
        if name in self._variables:
            del self._variables[name]
            return True
        return False

    def get_variable_info(self, name: str) -> Optional[Variable]:
        """Get information about a variable."""
        return self._variables.get(name)

    def list_variables(self) -> List[Variable]:
        """List all tracked variables."""
        return list(self._variables.values())

    def add_artifact(
        self,
        name: str,
        artifact_type: str,
        path: Optional[Path] = None,
        data: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DataArtifact:
        """
        Add a data artifact.

        Args:
            name: Artifact name
            artifact_type: Type of artifact
            path: Optional file path
            data: Optional in-memory data
            metadata: Optional metadata

        Returns:
            Created artifact
        """
        artifact = DataArtifact(
            name=name,
            artifact_type=artifact_type,
            path=path,
            data=data,
            metadata=metadata or {},
        )
        self._artifacts[name] = artifact
        return artifact

    def get_artifact(self, name: str) -> Optional[DataArtifact]:
        """Get an artifact by name."""
        return self._artifacts.get(name)

    def list_artifacts(self) -> List[DataArtifact]:
        """List all artifacts."""
        return list(self._artifacts.values())

    def record_execution(
        self,
        code: str,
        result: Any,
        success: bool,
        variables_created: Optional[List[str]] = None,
        variables_modified: Optional[List[str]] = None,
        artifacts_created: Optional[List[str]] = None,
        error: Optional[str] = None,
    ) -> ExecutionContext:
        """
        Record an execution in history.

        Args:
            code: Executed code
            result: Execution result
            success: Whether execution succeeded
            variables_created: New variables created
            variables_modified: Existing variables modified
            artifacts_created: New artifacts created
            error: Error message if failed

        Returns:
            ExecutionContext entry
        """
        context = ExecutionContext(
            code=code,
            result=result,
            success=success,
            timestamp=datetime.now(),
            variables_created=variables_created or [],
            variables_modified=variables_modified or [],
            artifacts_created=artifacts_created or [],
            error=error,
        )
        self._history.append(context)
        return context

    def get_history(self, limit: Optional[int] = None) -> List[ExecutionContext]:
        """
        Get execution history.

        Args:
            limit: Maximum number of entries (most recent)

        Returns:
            List of execution contexts
        """
        if limit is None:
            return self._history.copy()
        return self._history[-limit:]

    def get_context_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current state for the agent.

        Returns:
            Dictionary with state summary
        """
        return {
            "variables": {
                name: {
                    "type": var.type_name,
                    "preview": var.value_preview,
                    "length": var.length,
                    "shape": var.shape,
                }
                for name, var in self._variables.items()
            },
            "artifacts": {
                name: {
                    "type": art.artifact_type,
                    "path": str(art.path) if art.path else None,
                }
                for name, art in self._artifacts.items()
            },
            "execution_count": len(self._history),
            "last_execution": (
                {
                    "success": self._history[-1].success,
                    "error": self._history[-1].error,
                }
                if self._history else None
            ),
        }

    def create_checkpoint(self) -> Dict[str, Any]:
        """
        Create a checkpoint of the current state.

        Note: This only saves metadata, not actual values.
        For full state persistence, the kernel's namespace would
        need to be serialized separately.

        Returns:
            Checkpoint data
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "variables": {
                name: var.to_dict()
                for name, var in self._variables.items()
            },
            "artifacts": {
                name: {
                    "type": art.artifact_type,
                    "path": str(art.path) if art.path else None,
                    "metadata": art.metadata,
                }
                for name, art in self._artifacts.items()
            },
            "history_length": len(self._history),
        }

    def save_checkpoint(self, path: Path) -> None:
        """Save checkpoint to file."""
        checkpoint = self.create_checkpoint()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=2)
        logger.info(f"Checkpoint saved to {path}")

    def clear(self) -> None:
        """Clear all state."""
        self._variables.clear()
        self._artifacts.clear()
        self._history.clear()
        logger.info("State cleared")
