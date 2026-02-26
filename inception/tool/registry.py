"""
Tool Registry for managing available tools.

Provides:
- Tool registration and lookup
- Category-based organization
- Persistence for generated tools
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from inception.tool.base import Tool, ToolSpec
from inception.provider.base import ToolDefinition

logger = logging.getLogger(__name__)


@dataclass
class ToolEntry:
    """Entry in the tool registry."""
    tool: Tool
    category: str
    tags: Set[str]
    is_builtin: bool = True
    usage_count: int = 0


class ToolRegistry:
    """
    Central registry for all available tools.

    Features:
    - Register tools by name
    - Query by category or tags
    - Track usage statistics
    - Persist generated tools
    """

    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize the registry.

        Args:
            storage_path: Path for persisting generated tools
        """
        self._tools: Dict[str, ToolEntry] = {}
        self._categories: Dict[str, Set[str]] = {}  # category -> tool names
        self._tags: Dict[str, Set[str]] = {}  # tag -> tool names
        self._storage_path = storage_path

    def register(
        self,
        tool: Tool,
        is_builtin: bool = True,
        override: bool = False,
    ) -> None:
        """
        Register a tool.

        Args:
            tool: The tool to register
            is_builtin: Whether this is a built-in tool
            override: Whether to override existing tool with same name
        """
        name = tool.spec.name

        if name in self._tools and not override:
            raise ValueError(f"Tool '{name}' is already registered")

        # Create entry
        entry = ToolEntry(
            tool=tool,
            category=tool.spec.category,
            tags=set(tool.spec.tags),
            is_builtin=is_builtin,
        )

        self._tools[name] = entry

        # Update category index
        if entry.category not in self._categories:
            self._categories[entry.category] = set()
        self._categories[entry.category].add(name)

        # Update tag index
        for tag in entry.tags:
            if tag not in self._tags:
                self._tags[tag] = set()
            self._tags[tag].add(name)

        logger.debug(f"Registered tool: {name} (category: {entry.category})")

    def unregister(self, name: str) -> bool:
        """
        Unregister a tool by name.

        Returns:
            True if tool was found and removed
        """
        if name not in self._tools:
            return False

        entry = self._tools[name]

        # Remove from category index
        if entry.category in self._categories:
            self._categories[entry.category].discard(name)

        # Remove from tag index
        for tag in entry.tags:
            if tag in self._tags:
                self._tags[tag].discard(name)

        del self._tools[name]
        logger.debug(f"Unregistered tool: {name}")
        return True

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        entry = self._tools.get(name)
        return entry.tool if entry else None

    def get_spec(self, name: str) -> Optional[ToolSpec]:
        """Get a tool's specification by name."""
        tool = self.get(name)
        return tool.spec if tool else None

    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def list_all(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def list_by_category(self, category: str) -> List[str]:
        """List tools in a category."""
        return list(self._categories.get(category, set()))

    def list_by_tag(self, tag: str) -> List[str]:
        """List tools with a specific tag."""
        return list(self._tags.get(tag, set()))

    def list_categories(self) -> List[str]:
        """List all categories."""
        return list(self._categories.keys())

    def list_tags(self) -> List[str]:
        """List all tags."""
        return list(self._tags.keys())

    def get_tool_definitions(
        self,
        names: Optional[List[str]] = None,
        category: Optional[str] = None,
    ) -> List[ToolDefinition]:
        """
        Get tool definitions for LLM consumption.

        Args:
            names: Specific tools to include (None for all)
            category: Filter by category

        Returns:
            List of ToolDefinition objects
        """
        if names is not None:
            tool_names = [n for n in names if n in self._tools]
        elif category is not None:
            tool_names = self.list_by_category(category)
        else:
            tool_names = self.list_all()

        return [
            self._tools[name].tool.spec.to_tool_definition()
            for name in tool_names
        ]

    def search(
        self,
        query: str,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[ToolSpec]:
        """
        Search for tools matching criteria.

        Args:
            query: Search query (matches name or description)
            category: Filter by category
            tags: Filter by tags (any match)

        Returns:
            List of matching tool specs
        """
        results = []
        query_lower = query.lower()

        for name, entry in self._tools.items():
            # Category filter
            if category and entry.category != category:
                continue

            # Tags filter
            if tags and not any(t in entry.tags for t in tags):
                continue

            # Query match
            spec = entry.tool.spec
            if (query_lower in spec.name.lower() or
                query_lower in spec.description.lower()):
                results.append(spec)

        return results

    def record_usage(self, name: str) -> None:
        """Record tool usage for statistics."""
        if name in self._tools:
            self._tools[name].usage_count += 1

    def get_usage_stats(self) -> Dict[str, int]:
        """Get usage statistics for all tools."""
        return {
            name: entry.usage_count
            for name, entry in self._tools.items()
        }

    # Persistence methods

    def save(self, path: Optional[Path] = None) -> None:
        """
        Save generated tools to storage.

        Only saves non-builtin tools that have source code.
        """
        path = path or self._storage_path
        if not path:
            logger.warning("No storage path configured for tool persistence")
            return

        path.parent.mkdir(parents=True, exist_ok=True)

        # Collect generated tools
        generated = {}
        for name, entry in self._tools.items():
            if not entry.is_builtin and entry.tool.spec.source_code:
                generated[name] = {
                    "spec": {
                        "name": entry.tool.spec.name,
                        "description": entry.tool.spec.description,
                        "category": entry.tool.spec.category,
                        "tags": list(entry.tool.spec.tags),
                        "version": entry.tool.spec.version,
                        "source_code": entry.tool.spec.source_code,
                    },
                    "usage_count": entry.usage_count,
                }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(generated, f, indent=2)

        logger.info(f"Saved {len(generated)} generated tools to {path}")

    def load(self, path: Optional[Path] = None, factory: Optional[Any] = None) -> int:
        """
        Load generated tools from storage.

        Args:
            path: Path to load from (uses storage_path if not provided)
            factory: ToolFactory instance to recreate tools

        Returns:
            Number of tools loaded
        """
        path = path or self._storage_path
        if not path or not path.exists():
            return 0

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        loaded_count = 0
        for name, tool_data in data.items():
            spec = tool_data.get("spec", {})
            source_code = spec.get("source_code")

            if not source_code:
                logger.warning(f"Skipping tool '{name}': no source code")
                continue

            if factory:
                # Use factory to recreate the tool
                success = factory.create_and_register(
                    name=spec.get("name", name),
                    description=spec.get("description", ""),
                    code=source_code,
                    category=spec.get("category", "generated"),
                    tags=spec.get("tags", []),
                )
                if success:
                    # Restore usage count
                    if name in self._tools:
                        self._tools[name].usage_count = tool_data.get("usage_count", 0)
                    loaded_count += 1
                    logger.info(f"Loaded tool: {name}")
                else:
                    logger.warning(f"Failed to recreate tool: {name}")
            else:
                logger.warning(f"No factory provided, cannot recreate tool: {name}")

        logger.info(f"Loaded {loaded_count} generated tools from {path}")
        return loaded_count

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __iter__(self):
        return iter(self._tools.values())


# Global registry instance
_global_registry: Optional[ToolRegistry] = None


def get_global_registry() -> ToolRegistry:
    """Get the global tool registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry


def set_global_registry(registry: ToolRegistry) -> None:
    """Set the global tool registry."""
    global _global_registry
    _global_registry = registry
