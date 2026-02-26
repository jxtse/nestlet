"""Built-in tools for Inception."""

from inception.tool.builtin.code_exec import CodeExecutionTool
from inception.tool.builtin.file_ops import ReadFileTool, WriteFileTool, ListDirectoryTool
from inception.tool.builtin.llm_call import LLMCallTool
from inception.tool.builtin.office_parser import (
    ParseWordTool,
    ParseExcelTool,
    ParsePowerPointTool,
    ParsePDFTool,
)
from inception.tool.builtin.web_search import WebSearchTool

__all__ = [
    "CodeExecutionTool",
    "ReadFileTool",
    "WriteFileTool",
    "ListDirectoryTool",
    "LLMCallTool",
    "ParseWordTool",
    "ParseExcelTool",
    "ParsePowerPointTool",
    "ParsePDFTool",
    "WebSearchTool",
]


def register_builtin_tools(registry, kernel=None, provider=None, workspace=None, settings=None):
    """
    Register all built-in tools with the registry.

    Args:
        registry: ToolRegistry to register tools with
        kernel: PythonKernel for code execution (optional)
        provider: LLM provider for llm_call tool (optional)
        workspace: Workspace path for file operations (optional)
        settings: Settings object for tool configuration (optional)
    """
    # Code execution
    if kernel:
        registry.register(CodeExecutionTool(kernel))

    # File operations
    if workspace:
        registry.register(ReadFileTool(workspace))
        registry.register(WriteFileTool(workspace))
        registry.register(ListDirectoryTool(workspace))

    # LLM call
    if provider:
        registry.register(LLMCallTool(provider))

    # Office document parsers (always register)
    registry.register(ParseWordTool())
    registry.register(ParseExcelTool())
    registry.register(ParsePowerPointTool())
    registry.register(ParsePDFTool())

    # Web search
    if settings and settings.web_search.enabled:
        registry.register(WebSearchTool(
            config=settings.web_search,
            provider=provider,
        ))
