"""
Configuration settings for Inception.

Manages all configuration including LLM providers, execution settings,
and security policies.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class ProviderType(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"


class ExecutionMode(str, Enum):
    """Code execution mode."""
    SANDBOX = "sandbox"  # Restricted execution
    TRUSTED = "trusted"  # Full access (for trusted environments)


@dataclass
class ProviderConfig:
    """LLM provider configuration."""
    type: ProviderType
    api_key: Optional[str] = None
    model: str = "gpt-4o"
    base_url: Optional[str] = None
    # Azure-specific
    azure_endpoint: Optional[str] = None
    azure_deployment: Optional[str] = None
    api_version: Optional[str] = None
    # Rate limiting
    max_retries: int = 3
    timeout: float = 60.0
    # Token limits
    max_tokens: int = 4096
    temperature: float = 0.7

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ProviderConfig:
        """Create from dictionary."""
        provider_type = data.get("type", "openai")
        if isinstance(provider_type, str):
            provider_type = ProviderType(provider_type)
        return cls(
            type=provider_type,
            api_key=data.get("api_key") or os.getenv(f"{provider_type.value.upper()}_API_KEY"),
            model=data.get("model", "gpt-4o"),
            base_url=data.get("base_url"),
            azure_endpoint=data.get("azure_endpoint"),
            azure_deployment=data.get("azure_deployment"),
            api_version=data.get("api_version", "2024-02-15-preview"),
            max_retries=data.get("max_retries", 3),
            timeout=data.get("timeout", 60.0),
            max_tokens=data.get("max_tokens", 4096),
            temperature=data.get("temperature", 0.7),
        )


# Default module lists (defined outside class for reference in from_dict)
# Allow common modules needed for file operations and data analysis
DEFAULT_ALLOWED_MODULES: List[str] = [
    # Core Python modules
    "math", "statistics", "collections", "itertools", "functools",
    "datetime", "json", "re", "string", "textwrap",
    # File system modules (needed for reading files, checking paths)
    "os", "sys", "pathlib", "shutil", "glob", "fnmatch",
    # Data analysis libraries
    "numpy", "pandas", "scipy", "openpyxl", "xlrd", "olefile",
    # Other useful modules
    "importlib", "platform", "io", "csv", "pickle",
]
# Only block network-related and truly dangerous modules
DEFAULT_BLOCKED_MODULES: List[str] = [
    "socket", "requests", "urllib", "http", "ftplib",
    "smtplib", "telnetlib", "asyncio.subprocess",
]


@dataclass
class ExecutionConfig:
    """Code execution configuration."""
    mode: ExecutionMode = ExecutionMode.SANDBOX
    timeout: float = 30.0  # seconds
    max_memory_mb: int = 512
    # Module restrictions
    allowed_modules: List[str] = field(default_factory=lambda: DEFAULT_ALLOWED_MODULES.copy())
    blocked_modules: List[str] = field(default_factory=lambda: DEFAULT_BLOCKED_MODULES.copy())
    # Working directory for file operations
    workspace_dir: Optional[Path] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExecutionConfig:
        """Create from dictionary."""
        mode = data.get("mode", "sandbox")
        if isinstance(mode, str):
            mode = ExecutionMode(mode)

        workspace = data.get("workspace_dir")
        if workspace:
            workspace = Path(workspace)

        return cls(
            mode=mode,
            timeout=data.get("timeout", 30.0),
            max_memory_mb=data.get("max_memory_mb", 512),
            allowed_modules=data.get("allowed_modules", DEFAULT_ALLOWED_MODULES.copy()),
            blocked_modules=data.get("blocked_modules", DEFAULT_BLOCKED_MODULES.copy()),
            workspace_dir=workspace,
        )


@dataclass
class MemoryConfig:
    """Memory system configuration."""
    max_conversation_turns: int = 50
    max_working_memory_items: int = 20
    # Long-term memory persistence
    persist_tools: bool = True
    tools_storage_path: Optional[Path] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MemoryConfig:
        """Create from dictionary."""
        tools_path = data.get("tools_storage_path")
        if tools_path:
            tools_path = Path(tools_path)

        return cls(
            max_conversation_turns=data.get("max_conversation_turns", 50),
            max_working_memory_items=data.get("max_working_memory_items", 20),
            persist_tools=data.get("persist_tools", True),
            tools_storage_path=tools_path,
        )


@dataclass
class WebSearchConfig:
    """Web search configuration."""
    enabled: bool = True
    backend: str = "tavily"  # "tavily" | "duckduckgo"
    tavily_api_key: Optional[str] = None
    default_max_results: int = 5
    deep_search_max_results: int = 10
    default_language: str = "en"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WebSearchConfig:
        """Create from dictionary."""
        # Support environment variable for API key
        tavily_api_key = data.get("tavily_api_key") or os.getenv("TAVILY_API_KEY")

        return cls(
            enabled=data.get("enabled", True),
            backend=data.get("backend", "tavily"),
            tavily_api_key=tavily_api_key,
            default_max_results=data.get("default_max_results", 5),
            deep_search_max_results=data.get("deep_search_max_results", 10),
            default_language=data.get("default_language", "en"),
        )


@dataclass
class Settings:
    """
    Main settings container for Inception.

    Can be initialized from:
    - Environment variables
    - YAML configuration file
    - Direct instantiation
    """
    provider: ProviderConfig = field(default_factory=lambda: ProviderConfig(type=ProviderType.OPENAI))
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    web_search: WebSearchConfig = field(default_factory=WebSearchConfig)
    # Agent settings
    agent_name: str = "Inception"
    verbose: bool = False
    debug: bool = False
    # Plugin directory
    plugins_dir: Optional[Path] = None

    @classmethod
    def from_yaml(cls, path: Path | str) -> Settings:
        """Load settings from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data or {})

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Settings:
        """Create from dictionary."""
        plugins_dir = data.get("plugins_dir")
        if plugins_dir:
            plugins_dir = Path(plugins_dir)

        return cls(
            provider=ProviderConfig.from_dict(data.get("provider", {})),
            execution=ExecutionConfig.from_dict(data.get("execution", {})),
            memory=MemoryConfig.from_dict(data.get("memory", {})),
            web_search=WebSearchConfig.from_dict(data.get("web_search", {})),
            agent_name=data.get("agent_name", "Inception"),
            verbose=data.get("verbose", False),
            debug=data.get("debug", False),
            plugins_dir=plugins_dir,
        )

    @classmethod
    def from_env(cls) -> Settings:
        """Create settings from environment variables."""
        # Determine provider type
        provider_type = os.getenv("INCEPTION_PROVIDER", "openai")

        provider_config = ProviderConfig(
            type=ProviderType(provider_type),
            api_key=os.getenv(f"{provider_type.upper()}_API_KEY"),
            model=os.getenv("INCEPTION_MODEL", "gpt-4o"),
            base_url=os.getenv("INCEPTION_BASE_URL"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            api_version=os.getenv("AZURE_API_VERSION", "2024-02-15-preview"),
        )

        execution_config = ExecutionConfig(
            mode=ExecutionMode(os.getenv("INCEPTION_EXEC_MODE", "sandbox")),
            timeout=float(os.getenv("INCEPTION_EXEC_TIMEOUT", "30")),
        )

        web_search_config = WebSearchConfig(
            enabled=os.getenv("INCEPTION_WEB_SEARCH_ENABLED", "true").lower() == "true",
            backend=os.getenv("INCEPTION_WEB_SEARCH_BACKEND", "tavily"),
            tavily_api_key=os.getenv("TAVILY_API_KEY"),
            default_language=os.getenv("INCEPTION_WEB_SEARCH_LANGUAGE", "en"),
        )

        return cls(
            provider=provider_config,
            execution=execution_config,
            web_search=web_search_config,
            verbose=os.getenv("INCEPTION_VERBOSE", "").lower() == "true",
            debug=os.getenv("INCEPTION_DEBUG", "").lower() == "true",
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for serialization)."""
        return {
            "provider": {
                "type": self.provider.type.value,
                "model": self.provider.model,
                "base_url": self.provider.base_url,
                "max_retries": self.provider.max_retries,
                "timeout": self.provider.timeout,
                "max_tokens": self.provider.max_tokens,
                "temperature": self.provider.temperature,
            },
            "execution": {
                "mode": self.execution.mode.value,
                "timeout": self.execution.timeout,
                "max_memory_mb": self.execution.max_memory_mb,
                "allowed_modules": self.execution.allowed_modules,
                "blocked_modules": self.execution.blocked_modules,
            },
            "memory": {
                "max_conversation_turns": self.memory.max_conversation_turns,
                "max_working_memory_items": self.memory.max_working_memory_items,
                "persist_tools": self.memory.persist_tools,
            },
            "web_search": {
                "enabled": self.web_search.enabled,
                "backend": self.web_search.backend,
                "default_max_results": self.web_search.default_max_results,
                "deep_search_max_results": self.web_search.deep_search_max_results,
                "default_language": self.web_search.default_language,
            },
            "agent_name": self.agent_name,
            "verbose": self.verbose,
            "debug": self.debug,
        }

    def save_yaml(self, path: Path | str) -> None:
        """Save settings to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False)
