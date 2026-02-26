"""
LLM Call Tool.

Allows the agent to make sub-calls to the LLM for specific tasks.
"""

from __future__ import annotations

import asyncio
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
from inception.provider.base import Message

if TYPE_CHECKING:
    from inception.provider.base import BaseProvider


class LLMCallTool(Tool):
    """
    Tool for making sub-calls to the LLM.

    Useful for:
    - Breaking down complex reasoning
    - Generating code or text
    - Analyzing data
    """

    # Rate limiting: minimum interval between calls (in seconds)
    MIN_CALL_INTERVAL = 1.0

    def __init__(self, provider: "BaseProvider"):
        """
        Initialize with an LLM provider.

        Args:
            provider: The LLM provider for making calls
        """
        self._provider = provider
        self._last_call_time: float = 0.0
        self._spec = ToolSpec(
            name="llm_call",
            description=(
                "Make a call to the LLM for a specific sub-task. "
                "Useful for generating code, analyzing text, or complex reasoning. "
                "Use this for tasks that benefit from natural language understanding."
            ),
            parameters={
                "prompt": ParameterSpec(
                    name="prompt",
                    type=ParameterType.STRING,
                    description="The prompt/instruction for the LLM",
                    required=True,
                ),
                "system_prompt": ParameterSpec(
                    name="system_prompt",
                    type=ParameterType.STRING,
                    description="Optional system prompt to set context",
                    required=False,
                ),
                "temperature": ParameterSpec(
                    name="temperature",
                    type=ParameterType.NUMBER,
                    description="Temperature for generation (0.0-2.0, default: 0.7)",
                    required=False,
                    default=0.7,
                ),
                "max_tokens": ParameterSpec(
                    name="max_tokens",
                    type=ParameterType.INTEGER,
                    description="Maximum tokens to generate",
                    required=False,
                    default=2048,
                ),
            },
            returns=ReturnSpec(
                type=ParameterType.STRING,
                description="The LLM's response",
            ),
            category="llm",
            tags=["llm", "generation", "reasoning"],
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    async def _rate_limit(self) -> None:
        """Apply rate limiting to avoid API throttling."""
        current_time = time.time()
        elapsed = current_time - self._last_call_time
        if elapsed < self.MIN_CALL_INTERVAL:
            await asyncio.sleep(self.MIN_CALL_INTERVAL - elapsed)
        self._last_call_time = time.time()

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Make an LLM call."""
        prompt = kwargs.get("prompt", "")
        system_prompt = kwargs.get("system_prompt")
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 2048)

        if not prompt:
            return ToolResult.fail("No prompt provided")

        try:
            # Apply rate limiting before making the call
            await self._rate_limit()

            messages = []

            if system_prompt:
                messages.append(Message.system(system_prompt))

            messages.append(Message.user(prompt))

            response = await self._provider.complete(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            return ToolResult.ok(result=response.content)

        except Exception as e:
            return ToolResult.fail(f"LLM call failed: {e}")


# Example prompts for common analysis tasks (for reference in tool description)
ANALYSIS_EXAMPLES = """
## Example Prompts for Common Tasks

### Sentiment Analysis
```
Analyze the sentiment of the following text. Return a JSON object with:
- 'sentiment': positive/negative/neutral/mixed
- 'confidence': 0.0-1.0
- 'explanation': brief reasoning

Text: {your_text}
```

### Summarization
```
Provide a concise summary of the following text in 2-3 sentences.
Focus on the main points and key takeaways.

Text: {your_text}
```

### Key Points Extraction
```
Extract the key points from the following text as a bullet list.
Prioritize actionable insights and important facts.

Text: {your_text}
```

### Named Entity Recognition
```
Extract named entities from the following text. Categorize them into:
- People, Organizations, Locations, Dates, Products, Events

Text: {your_text}
```

### Topic Identification
```
Identify the main topics discussed in the following text.
For each topic, provide a relevance score (0-1) and brief description.

Text: {your_text}
```

### Custom Analysis
```
{your_custom_instruction}

Text: {your_text}
```
"""


class AnalyzeTextTool(Tool):
    """
    Tool for analyzing text using LLM capabilities.

    This tool provides full flexibility for the agent to craft custom prompts
    for any type of text analysis. The agent should write appropriate prompts
    based on the specific analysis needs.
    """

    # Rate limiting: minimum interval between calls (in seconds)
    MIN_CALL_INTERVAL = 1.0

    def __init__(self, provider: "BaseProvider"):
        """
        Initialize with an LLM provider.

        Args:
            provider: The LLM provider for making calls
        """
        self._provider = provider
        self._last_call_time: float = 0.0
        self._spec = ToolSpec(
            name="analyze_text",
            description=(
                "Analyze text using a custom prompt. The agent has full control over "
                "the analysis approach by crafting appropriate prompts.\n\n"
                "Common use cases include: sentiment analysis, summarization, "
                "key points extraction, entity recognition, topic identification, "
                "translation, paraphrasing, fact-checking, and any custom analysis.\n\n"
                f"{ANALYSIS_EXAMPLES}"
            ),
            parameters={
                "text": ParameterSpec(
                    name="text",
                    type=ParameterType.STRING,
                    description="The text to analyze",
                    required=True,
                ),
                "instruction": ParameterSpec(
                    name="instruction",
                    type=ParameterType.STRING,
                    description=(
                        "Custom instruction describing what analysis to perform. "
                        "Be specific about the desired output format (e.g., JSON, bullet points, prose). "
                        "Examples: 'Summarize in 3 sentences', 'Extract all dates mentioned', "
                        "'Identify the author\\'s tone and provide evidence'"
                    ),
                    required=True,
                ),
                "output_format": ParameterSpec(
                    name="output_format",
                    type=ParameterType.STRING,
                    description=(
                        "Desired output format: 'json', 'text', 'markdown', 'list'. "
                        "Default is 'text'. If 'json', the tool will attempt to parse the response."
                    ),
                    required=False,
                    default="text",
                ),
                "temperature": ParameterSpec(
                    name="temperature",
                    type=ParameterType.NUMBER,
                    description=(
                        "Temperature for generation (0.0-2.0). "
                        "Lower values (0.1-0.3) for factual/consistent output, "
                        "higher values (0.7-1.0) for creative analysis. Default: 0.3"
                    ),
                    required=False,
                    default=0.3,
                ),
            },
            returns=ReturnSpec(
                type=ParameterType.OBJECT,
                description="Analysis results (parsed JSON if output_format='json', otherwise raw text)",
            ),
            category="llm",
            tags=["llm", "analysis", "text"],
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    async def _rate_limit(self) -> None:
        """Apply rate limiting to avoid API throttling."""
        current_time = time.time()
        elapsed = current_time - self._last_call_time
        if elapsed < self.MIN_CALL_INTERVAL:
            await asyncio.sleep(self.MIN_CALL_INTERVAL - elapsed)
        self._last_call_time = time.time()

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Analyze text with custom instruction."""
        text = kwargs.get("text", "")
        instruction = kwargs.get("instruction", "")
        output_format = kwargs.get("output_format", "text")
        temperature = kwargs.get("temperature", 0.3)

        if not text:
            return ToolResult.fail("No text provided")

        if not instruction:
            return ToolResult.fail("No instruction provided. Please specify what analysis to perform.")

        # Build the prompt
        prompt = f"{instruction}\n\nText:\n{text}"

        # Build system prompt based on output format
        system_prompts = {
            "json": "You are a text analysis assistant. Respond with valid JSON only, no additional text.",
            "markdown": "You are a text analysis assistant. Format your response in clean Markdown.",
            "list": "You are a text analysis assistant. Format your response as a clear bullet-point list.",
            "text": "You are a text analysis assistant. Provide clear and concise analysis.",
        }
        system_prompt = system_prompts.get(output_format, system_prompts["text"])

        try:
            # Apply rate limiting before making the call
            await self._rate_limit()

            messages = [
                Message.system(system_prompt),
                Message.user(prompt),
            ]

            response = await self._provider.complete(
                messages=messages,
                temperature=temperature,
            )

            # Process response based on output format
            if output_format == "json":
                import json
                try:
                    # Try to extract JSON from response (handle markdown code blocks)
                    content = response.content.strip()
                    if content.startswith("```"):
                        # Extract content from code block
                        lines = content.split("\n")
                        content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
                    result = json.loads(content)
                    return ToolResult.ok(result=result)
                except json.JSONDecodeError:
                    # Return raw response if JSON parsing fails
                    return ToolResult.ok(result={"raw_response": response.content, "parse_error": "Failed to parse as JSON"})
            else:
                return ToolResult.ok(result=response.content)

        except Exception as e:
            return ToolResult.fail(f"Analysis failed: {e}")
