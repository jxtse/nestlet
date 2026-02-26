"""
Hybrid Agent - Main agent implementation.

Combines neural (LLM) and symbolic (code execution) computation
with autonomous tool-making capabilities.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from inception.agent.base import (
    Action,
    ActionResult,
    ActionType,
    BaseAgent,
    Context,
    ThinkResult,
)
from inception.config.settings import Settings
from inception.executor.kernel import PythonKernel
from inception.executor.state import StateManager
from inception.memory.conversation import ConversationMemory
from inception.memory.working_memory import MemoryItemType, WorkingMemory
from inception.provider.base import BaseProvider, Message, ToolCall
from inception.provider.openai import OpenAIProvider
from inception.tool.base import Tool, ToolResult
from inception.tool.factory import ToolFactory
from inception.tool.registry import ToolRegistry
from inception.metacognition.capability import CapabilityAssessor
from inception.metacognition.decision import ComputationDecider, EnhancedDecision
from inception.metacognition.data_estimator import DataSizeEstimator

logger = logging.getLogger(__name__)
_console = Console()


# System prompt for the hybrid agent
SYSTEM_PROMPT = """You are Inception, a neuro-symbolic AI agent that combines neural reasoning (LLM) with symbolic computation (code execution).

## Your Capabilities

1. **Neural Reasoning**: You can reason, plan, analyze, and generate ideas using natural language.

2. **Symbolic Computation**: You can execute Python code for precise calculations, data processing, and algorithmic tasks.

3. **Tool Usage**: You have access to various tools for specific tasks.

4. **Tool Creation**: When existing tools are insufficient, you can create new tools.

## Cognitive Adaptation Principles

Choose computation mode based on the cognitive nature of the task:

| Cognitive Nature | Best Mode | Reason |
|-----------------|-----------|--------|
| Semantic Understanding (topics, sentiment, intent) | Neural | LLM excels at meaning, context, nuance |
| Precise Computation (math, aggregation) | Symbolic | Code guarantees accuracy |
| Creative Generation (writing, ideation) | Neural | LLM excels at novel combinations |
| Deterministic Iteration (loops, filtering) | Symbolic | Loops don't hallucinate |
| Pattern Discovery (open-ended analysis) | Neural | LLM can discover without predefined rules |
| Data Transformation (ETL, formatting) | Symbolic | Code is reliable and reproducible |

## Scale-Aware Principles

**CRITICAL: When data scale exceeds working memory capacity, you MUST batch process.**

1. **Detect**: Estimate data scale (row count, token count) BEFORE processing
2. **Chunk**: Use Symbolic to split data into manageable batches
3. **Process**: Apply Neural analysis to each batch
4. **Aggregate**: Use Symbolic to merge results

**Token Limits**: Never send more than ~4000 tokens of data in a single prompt.
- 100 short items (~30 tokens each) = ~3000 tokens ✓
- 1000 items = MUST batch into groups of ~100

## Strength Delegation Principles

**Use Neural Computing for:**
- Topic identification and theme discovery
- Sentiment analysis and emotion detection
- Content summarization and abstraction
- Intent recognition and classification
- Semantic similarity and meaning comparison
- Open-ended analysis without predefined categories

**Use Symbolic Computing for:**
- Counting, summing, averaging
- Sorting, filtering, grouping
- Loop iteration and batch processing
- File I/O and data loading
- String manipulation and formatting
- Deterministic transformations

## Critical Rules

1. **NEVER send large datasets directly in prompts** - MUST chunk first
2. **NEVER hardcode topic/theme lists** - use Neural for semantic discovery
3. **ALWAYS estimate data scale** before processing
4. **For semantic tasks on large data**: Use Symbolic to batch + Neural to analyze each batch

## Python Environment

This project uses **uv** as the Python package manager (NOT pip).

**IMPORTANT: When you need to install Python packages:**
- ❌ DO NOT use: `pip install <package>` or `subprocess.run(["pip", ...])`
- ✅ USE: `uv pip install <package>` or `uv add <package>`

Example commands:
```bash
# Install a package
uv pip install pandas

# Install multiple packages
uv pip install numpy scipy matplotlib

# Add a package to project dependencies
uv add requests
```

## Response Format

When you need to execute code, wrap it in a code block:
```python
# Your code here
```

When you have a final answer, provide it clearly.
"""


class HybridAgent(BaseAgent):
    """
    The main hybrid agent that orchestrates neural and symbolic computation.

    Features:
    - Intelligent mode selection (neuro vs symbolic)
    - Stateful code execution
    - Autonomous tool creation
    - Memory management
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        provider: Optional[BaseProvider] = None,
        kernel: Optional[PythonKernel] = None,
        registry: Optional[ToolRegistry] = None,
    ):
        """
        Initialize the hybrid agent.

        Args:
            settings: Configuration settings
            provider: LLM provider (created from settings if not provided)
            kernel: Python kernel (created if not provided)
            registry: Tool registry (created if not provided)
        """
        self._settings = settings or Settings.from_env()

        # Initialize provider
        if provider:
            self._provider = provider
        else:
            self._provider = OpenAIProvider(self._settings.provider)

        # Initialize kernel with allowed/blocked modules from settings
        self._kernel = kernel or PythonKernel(
            timeout=self._settings.execution.timeout,
            allowed_modules=set(self._settings.execution.allowed_modules)
            if self._settings.execution.allowed_modules
            else None,
            blocked_modules=set(self._settings.execution.blocked_modules)
            if self._settings.execution.blocked_modules
            else None,
        )

        # Initialize registry and factory with persistence
        tools_storage_path = None
        if self._settings.memory.persist_tools:
            # Use configured path or default to plugins/generated_tools.json
            if self._settings.memory.tools_storage_path:
                tools_storage_path = self._settings.memory.tools_storage_path
            elif self._settings.plugins_dir:
                tools_storage_path = self._settings.plugins_dir / "generated_tools.json"
            else:
                # Default to plugins directory relative to current working directory
                tools_storage_path = Path("plugins/generated_tools.json")

        self._registry = registry or ToolRegistry(storage_path=tools_storage_path)
        self._factory = ToolFactory(self._registry)

        # Initialize memory
        self._conversation = ConversationMemory(
            max_turns=self._settings.memory.max_conversation_turns,
            system_prompt=SYSTEM_PROMPT,
        )
        self._working_memory = WorkingMemory(
            max_items=self._settings.memory.max_working_memory_items,
        )
        self._state = StateManager()

        # Initialize metacognition components
        self._assessor = CapabilityAssessor()
        self._data_estimator = DataSizeEstimator()
        self._decider = ComputationDecider(
            assessor=self._assessor,
            data_estimator=self._data_estimator,
        )

        # Initialize state
        self._initialized = False

        # Verbose mode for debugging
        self._verbose = getattr(settings, "verbose", False) if settings else False

    @property
    def name(self) -> str:
        return self._settings.agent_name

    @property
    def description(self) -> str:
        return "Neuro-symbolic agent combining LLM reasoning with code execution"

    async def initialize(self) -> None:
        """Initialize the agent components."""
        if self._initialized:
            return

        # Initialize kernel
        await self._kernel.initialize()

        # Register built-in tools
        from inception.tool.builtin import register_builtin_tools

        register_builtin_tools(
            self._registry,
            kernel=self._kernel,
            provider=self._provider,
            settings=self._settings,
        )

        # Load persisted generated tools
        if self._settings.memory.persist_tools and self._registry._storage_path:
            loaded = self._registry.load(factory=self._factory)
            if loaded > 0:
                logger.info(f"Loaded {loaded} persisted tools")

        self._initialized = True
        logger.info(f"Agent '{self.name}' initialized")

    async def think(self, input: str, context: Context) -> ThinkResult:
        """
        Think about the input and decide on an action.

        Uses the LLM to reason about the task and choose an approach.
        Integrates metacognition for intelligent mode selection.
        """
        # === Pre-think Metacognition Assessment ===
        state_summary = self._state.get_context_summary()
        exec_context = {"variables": state_summary.get("variables", {})}
        meta_decision = self._decider.decide(input, exec_context)

        # Build metacognition guidance
        guidance = self._build_metacognition_guidance(meta_decision)

        # Build prompt with context and guidance
        prompt_parts = [input]

        # Insert metacognition guidance if available
        if guidance:
            prompt_parts.insert(0, f"## Processing Guidance\n{guidance}\n---\n")

        if context.to_prompt_context():
            prompt_parts.insert(0, f"Context:\n{context.to_prompt_context()}\n")

        user_message = "\n".join(prompt_parts)

        # Get available tools
        tool_definitions = self._registry.get_tool_definitions()

        # Call LLM
        messages = self._conversation.get_messages()
        messages.append(Message.user(user_message))

        response = await self._provider.complete(
            messages=messages,
            tools=tool_definitions if tool_definitions else None,
            tool_choice="auto" if tool_definitions else None,
        )

        # Parse response to determine action
        action = self._parse_response_to_action(response.content, response.tool_calls)

        # Create think result
        return ThinkResult(
            reasoning=response.content,
            action=action,
            confidence=0.8,  # Could be refined with more sophisticated analysis
        )

    def _parse_response_to_action(
        self,
        content: str,
        tool_calls: List[ToolCall],
    ) -> Action:
        """Parse LLM response to determine the action to take."""
        # Check for tool calls
        if tool_calls:
            tc = tool_calls[0]  # Take first tool call
            return Action.tool_call(tc.name, tc.arguments)

        # Check for code blocks
        code_match = re.search(r"```python\n(.*?)```", content, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
            return Action.code_exec(code)

        # Check for tool creation request
        if "create_tool" in content.lower() or "new tool" in content.lower():
            # Parse tool creation details
            tool_match = re.search(
                r"Tool Name:\s*(\w+).*?Description:\s*(.+?)(?:Code:|```)",
                content,
                re.DOTALL | re.IGNORECASE,
            )
            if tool_match:
                return Action.create_tool(
                    name=tool_match.group(1).strip(),
                    description=tool_match.group(2).strip(),
                    code=code_match.group(1) if code_match else "",
                )

        # Default to response
        return Action.respond(content)

    def _build_metacognition_guidance(self, decision: EnhancedDecision) -> str:
        """
        Build metacognition guidance for the LLM based on the decision.

        Args:
            decision: Enhanced decision from the metacognition system

        Returns:
            Formatted guidance string to prepend to the prompt
        """
        # Skip guidance for trivial tasks
        if (
            not decision.warnings
            and not decision.requires_semantic
            and decision.data_estimate
            and not decision.data_estimate.requires_batching()
        ):
            return ""

        lines = []

        # Add mode recommendation
        lines.append(f"**Recommended Mode**: {decision.mode.value.upper()}")

        # Add reasoning
        if decision.reasoning:
            lines.append(f"**Reason**: {decision.reasoning}")

        # Add warnings (most important)
        if decision.warnings:
            lines.append("")
            for warning in decision.warnings:
                lines.append(f"⚠️ **WARNING**: {warning}")

        # Add data scale information
        if decision.data_estimate and decision.data_estimate.requires_batching():
            lines.append("")
            lines.append(f"**Data Scale**: {decision.data_estimate.scale.value.upper()}")
            lines.append(f"**Estimated Items**: ~{decision.data_estimate.estimated_items:,}")
            lines.append(f"**Strategy**: {decision.processing_strategy.value}")
            lines.append(f"**Recommended Batch Size**: {decision.recommended_batch_size}")

        # Add semantic task guidance
        if decision.requires_semantic:
            lines.append("")
            if not decision.is_enumerable:
                lines.append("**Semantic Task Type**: Open-ended discovery")
                lines.append("  - Do NOT hardcode topic/theme lists")
                lines.append("  - Use llm_call for semantic discovery")
            else:
                lines.append("**Semantic Task Type**: Closed-set classification")

        # Add orchestration plan
        if decision.orchestration_plan:
            lines.append("")
            lines.append("**Orchestration Plan**:")
            for i, step in enumerate(decision.orchestration_plan, 1):
                mode = step.get("mode", "unknown")
                purpose = step.get("purpose", "")
                lines.append(f"  {i}. [{mode}] {purpose}")

        return "\n".join(lines)

    async def act(self, action: Action, context: Context) -> ActionResult:
        """Execute an action."""
        start_time = time.time()

        try:
            if action.type == ActionType.TOOL_CALL:
                return await self._execute_tool(action)

            elif action.type == ActionType.CODE_EXEC:
                return await self._execute_code(action)

            elif action.type == ActionType.CREATE_TOOL:
                return await self._create_tool(action)

            elif action.type == ActionType.RESPOND:
                return ActionResult.ok(
                    result=action.response,
                    task_complete=True,
                    execution_time=time.time() - start_time,
                )

            else:
                return ActionResult.fail(
                    f"Unknown action type: {action.type}",
                    execution_time=time.time() - start_time,
                )

        except Exception as e:
            logger.exception(f"Action execution failed: {e}")
            return ActionResult.fail(
                str(e),
                execution_time=time.time() - start_time,
            )

    async def _execute_tool(self, action: Action) -> ActionResult:
        """Execute a tool call."""
        tool_name = action.tool_name
        tool_args = action.tool_args or {}

        if not tool_name:
            return ActionResult.fail("No tool name specified")

        tool = self._registry.get(tool_name)
        if not tool:
            return ActionResult.fail(f"Tool not found: {tool_name}")

        # Verbose output: show tool call
        if self._verbose:
            _console.print(
                Panel(
                    f"[bold cyan]Tool:[/bold cyan] {tool_name}\n"
                    f"[bold cyan]Arguments:[/bold cyan] {json.dumps(tool_args, indent=2, ensure_ascii=False)}",
                    title="🔧 Tool Call",
                    border_style="cyan",
                )
            )

        # Execute tool
        result = await tool(**tool_args)

        # Verbose output: show tool result
        if self._verbose:
            if result.success:
                _console.print(
                    Panel(
                        str(result.result)[:2000],
                        title="✅ Tool Result",
                        border_style="green",
                    )
                )
            else:
                _console.print(
                    Panel(
                        f"[red]{result.error}[/red]",
                        title="❌ Tool Error",
                        border_style="red",
                    )
                )

        # Record usage
        self._registry.record_usage(tool_name)

        # Update working memory
        if result.success:
            self._working_memory.add_result(
                content=result.result,
                name=f"tool_{tool_name}_result",
            )
        else:
            self._working_memory.add_error(
                error=result.error or "Unknown error",
                error_type="ToolError",
            )

        return ActionResult(
            success=result.success,
            result=result.result,
            error=result.error,
            tool_output=str(result.result) if result.success else result.error,
            execution_time=result.execution_time,
        )

    async def _execute_code(self, action: Action) -> ActionResult:
        """Execute Python code."""
        code = action.code
        if not code:
            return ActionResult.fail("No code specified")

        # Verbose output: show code being executed
        if self._verbose:
            _console.print(
                Panel(
                    Syntax(code, "python", theme="monokai", line_numbers=True),
                    title="🐍 Executing Code",
                    border_style="yellow",
                )
            )

        # Execute in kernel
        result = await self._kernel.execute(code)

        # Verbose output: show execution result
        if self._verbose:
            if result.success:
                output_text = result.output or str(result.result) or "(no output)"
                _console.print(
                    Panel(
                        output_text[:2000],
                        title="✅ Code Output",
                        border_style="green",
                    )
                )
            else:
                _console.print(
                    Panel(
                        f"[red]{result.error}[/red]",
                        title="❌ Execution Error",
                        border_style="red",
                    )
                )

        # Update state tracking
        if result.success:
            for var_name in result.variables_created:
                value = self._kernel.get_variable(var_name)
                if value is not None:
                    self._state.update_variable(var_name, value, is_new=True)

            self._working_memory.add_result(
                content=result.output or result.result,
                name="code_result",
            )
        else:
            self._working_memory.add_error(
                error=result.error or "Code execution failed",
                error_type=result.error_type,
            )

        return ActionResult(
            success=result.success,
            result=result.result,
            error=result.error,
            code_output=result.output,
            variables_created=result.variables_created,
            execution_time=0.0,
        )

    async def _create_tool(self, action: Action) -> ActionResult:
        """Create a new tool."""
        metadata = action.metadata
        tool_name = metadata.get("tool_name", "")
        tool_description = metadata.get("tool_description", "")
        tool_code = metadata.get("tool_code", "")

        if not tool_name or not tool_code:
            return ActionResult.fail("Tool name and code are required")

        # Verbose output: show tool creation
        if self._verbose:
            _console.print(
                Panel(
                    f"[bold magenta]Name:[/bold magenta] {tool_name}\n"
                    f"[bold magenta]Description:[/bold magenta] {tool_description}",
                    title="🛠️ Creating Tool",
                    border_style="magenta",
                )
            )

        # Create tool using factory
        success = self._factory.create_and_register(
            name=tool_name,
            description=tool_description,
            code=tool_code,
        )

        if success:
            # Save tools to persist the new tool
            self._save_tools()

            self._working_memory.add_observation(
                content=f"Created new tool: {tool_name}",
                name="tool_creation",
            )

            if self._verbose:
                _console.print(
                    Panel(
                        f"Tool '{tool_name}' created and saved successfully!",
                        title="✅ Tool Created",
                        border_style="green",
                    )
                )

            return ActionResult.ok(
                result=f"Successfully created tool: {tool_name}",
            )
        else:
            if self._verbose:
                _console.print(
                    Panel(
                        f"[red]Failed to create tool: {tool_name}[/red]",
                        title="❌ Tool Creation Failed",
                        border_style="red",
                    )
                )
            return ActionResult.fail(
                f"Failed to create tool: {tool_name}",
            )

    def _save_tools(self) -> None:
        """Save generated tools to persistent storage."""
        if self._settings.memory.persist_tools and self._registry._storage_path:
            try:
                self._registry.save()
                logger.debug("Saved generated tools to storage")
            except Exception as e:
                logger.warning(f"Failed to save tools: {e}")

    async def run(
        self,
        input: str,
        context: Optional[Context] = None,
        max_iterations: int = 10,
        images: Optional[List[str]] = None,
    ) -> str:
        """
        Run the agent on an input.

        Args:
            input: User input
            context: Optional initial context
            max_iterations: Maximum iterations
            images: Optional list of image paths or URLs

        Returns:
            Final response
        """
        # Ensure initialized
        if not self._initialized:
            await self.initialize()

        # Set up context
        if context is None:
            context = Context(
                user_input=input,
                available_tools=self._registry.list_all(),
                execution_state=self._state.get_context_summary(),
                working_memory=self._working_memory.get_context_summary(),
            )

        # Set current task
        self._working_memory.set_current_task(input)

        # Process images if provided
        image_data = None
        if images:
            image_data = self._process_images(images)

        # Add user message to conversation
        self._conversation.add_user_message(input, images=image_data)

        iterations = 0
        responses = []

        while iterations < max_iterations:
            # Think
            think_result = await self.think(input, context)

            logger.debug(f"Iteration {iterations + 1}: {think_result.action.type}")

            # Record reasoning
            self._working_memory.add_observation(
                content=think_result.reasoning[:200],
                name=f"reasoning_{iterations}",
            )

            # Act
            action_result = await self.act(think_result.action, context)

            # Handle response action
            if think_result.action.type == ActionType.RESPOND:
                response = think_result.action.response or ""
                self._conversation.add_assistant_message(response)
                return response

            # Handle completion
            if action_result.task_complete:
                response = str(action_result.result)
                self._conversation.add_assistant_message(response)
                return response

            # Handle tool call results
            if think_result.action.type == ActionType.TOOL_CALL:
                # Add assistant message with tool call
                self._conversation.add_assistant_message(
                    content=think_result.reasoning,
                    tool_calls=[
                        {
                            "id": f"call_{iterations}",
                            "name": think_result.action.tool_name,
                            "arguments": think_result.action.tool_args or {},
                        }
                    ],
                )
                # Add tool result
                self._conversation.add_tool_result(
                    tool_call_id=f"call_{iterations}",
                    name=think_result.action.tool_name or "unknown",
                    result=action_result.tool_output or str(action_result.result),
                )

            # Handle code execution results
            if think_result.action.type == ActionType.CODE_EXEC:
                code_result = action_result.code_output or str(action_result.result)
                self._conversation.add_assistant_message(
                    content=f"{think_result.reasoning}\n\nExecution result:\n{code_result}",
                )

            # Update context for next iteration
            context = Context(
                user_input=input,
                available_tools=self._registry.list_all(),
                execution_state=self._state.get_context_summary(),
                working_memory=self._working_memory.get_context_summary(),
            )

            iterations += 1

        # Max iterations reached
        return "I've reached the maximum number of iterations. Here's what I found:\n" + str(
            self._working_memory.get_context_summary()
        )

    async def chat(self, message: str, images: Optional[List[str]] = None) -> str:
        """
        Convenience method for single-turn chat.

        Args:
            message: User message
            images: Optional list of image paths or URLs

        Returns:
            Agent response
        """
        return await self.run(message, images=images)

    def _process_images(self, images: List[str]) -> List[Dict[str, Any]]:
        """
        Process image paths/URLs into image data for the conversation.

        Args:
            images: List of image paths or URLs

        Returns:
            List of image data dictionaries
        """
        import base64

        result = []
        for img in images:
            if img.startswith(("http://", "https://")):
                # URL
                result.append({"url": img})
            else:
                # File path
                path = Path(img)
                if not path.exists():
                    logger.warning(f"Image not found: {img}")
                    continue

                # Determine media type
                suffix = path.suffix.lower()
                media_types = {
                    ".png": "image/png",
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".gif": "image/gif",
                    ".webp": "image/webp",
                }
                media_type = media_types.get(suffix, "image/png")

                # Read and encode
                with open(path, "rb") as f:
                    base64_data = base64.b64encode(f.read()).decode("utf-8")

                result.append(
                    {
                        "base64_data": base64_data,
                        "media_type": media_type,
                    }
                )

        return result

    def reset(self) -> None:
        """Reset the agent state."""
        # Save tools before reset
        self._save_tools()

        self._conversation.clear()
        self._working_memory.clear()
        self._state.clear()
        self._kernel.reset()
        logger.info("Agent state reset")

    def shutdown(self) -> None:
        """Shutdown the agent and save state."""
        # Save generated tools
        self._save_tools()
        logger.info("Agent shutdown complete")
