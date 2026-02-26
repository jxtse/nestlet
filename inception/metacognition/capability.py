"""
Capability Assessment.

Evaluates the agent's capabilities for different types of tasks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class TaskType(str, Enum):
    """Types of tasks the agent can handle."""
    COMPUTATION = "computation"  # Mathematical/numerical tasks
    DATA_ANALYSIS = "data_analysis"  # Data processing and analysis
    REASONING = "reasoning"  # Logical reasoning
    CREATIVE = "creative"  # Creative generation
    INFORMATION = "information"  # Information retrieval/synthesis
    CODE_GENERATION = "code_generation"  # Writing code
    CONVERSATION = "conversation"  # General conversation
    UNKNOWN = "unknown"


@dataclass
class TaskCharacteristics:
    """Characteristics of a task."""
    task_type: TaskType
    requires_precision: bool = False
    requires_iteration: bool = False
    requires_external_data: bool = False
    requires_creativity: bool = False
    complexity_score: float = 0.5  # 0-1
    # Detected entities
    numbers: List[str] = field(default_factory=list)
    operations: List[str] = field(default_factory=list)
    data_references: List[str] = field(default_factory=list)
    # Suggested approach
    suggested_tools: List[str] = field(default_factory=list)


class CapabilityAssessor:
    """
    Assesses the agent's capability for different tasks.

    Helps decide whether to use neural or symbolic computation.
    """

    # Patterns for task type detection
    COMPUTATION_PATTERNS = [
        r"calculate", r"compute", r"sum", r"average", r"mean",
        r"multiply", r"divide", r"add", r"subtract",
        r"\d+\s*[\+\-\*\/\^]\s*\d+",  # Math expressions
        r"what is \d+", r"how much is",
    ]

    DATA_ANALYSIS_PATTERNS = [
        r"analyze", r"analysis", r"data", r"dataset",
        r"statistics", r"correlation", r"distribution",
        r"csv", r"excel", r"dataframe", r"table",
        r"plot", r"chart", r"graph", r"visualize",
    ]

    REASONING_PATTERNS = [
        r"why", r"explain", r"reason", r"logic",
        r"deduce", r"infer", r"conclude", r"because",
        r"if.*then", r"therefore", r"thus",
    ]

    CREATIVE_PATTERNS = [
        r"write", r"create", r"generate", r"compose",
        r"story", r"poem", r"essay", r"creative",
        r"imagine", r"design", r"brainstorm",
    ]

    CODE_PATTERNS = [
        r"code", r"function", r"program", r"script",
        r"implement", r"algorithm", r"class", r"method",
        r"python", r"javascript", r"programming",
    ]

    # Operations that need symbolic computation
    SYMBOLIC_OPERATIONS = {
        "+", "-", "*", "/", "^", "**", "%",
        "sqrt", "sin", "cos", "tan", "log", "exp",
        "sum", "mean", "median", "std", "var",
    }

    def __init__(self):
        """Initialize the assessor."""
        self._history: List[Dict[str, Any]] = []

    def assess(self, task: str) -> TaskCharacteristics:
        """
        Assess a task and determine its characteristics.

        Args:
            task: Task description

        Returns:
            TaskCharacteristics with analysis
        """
        task_lower = task.lower()

        # Detect task type
        task_type = self._detect_task_type(task_lower)

        # Extract numbers
        numbers = re.findall(r"\d+(?:\.\d+)?", task)

        # Extract operations
        operations = self._extract_operations(task_lower)

        # Extract data references
        data_refs = self._extract_data_references(task_lower)

        # Determine characteristics
        requires_precision = (
            task_type in (TaskType.COMPUTATION, TaskType.DATA_ANALYSIS) or
            len(numbers) > 2 or
            len(operations) > 0
        )

        requires_iteration = any(
            word in task_lower
            for word in ["each", "every", "all", "iterate", "loop", "for each"]
        )

        requires_external_data = bool(data_refs) or any(
            word in task_lower
            for word in ["file", "url", "api", "database", "fetch"]
        )

        requires_creativity = task_type in (TaskType.CREATIVE,)

        # Estimate complexity
        complexity = self._estimate_complexity(
            task,
            numbers,
            operations,
            data_refs,
        )

        # Suggest tools
        suggested_tools = self._suggest_tools(
            task_type,
            requires_precision,
            requires_external_data,
        )

        characteristics = TaskCharacteristics(
            task_type=task_type,
            requires_precision=requires_precision,
            requires_iteration=requires_iteration,
            requires_external_data=requires_external_data,
            requires_creativity=requires_creativity,
            complexity_score=complexity,
            numbers=numbers,
            operations=operations,
            data_references=data_refs,
            suggested_tools=suggested_tools,
        )

        # Record for learning
        self._history.append({
            "task": task,
            "characteristics": characteristics,
        })

        return characteristics

    def _detect_task_type(self, task: str) -> TaskType:
        """Detect the primary task type."""
        scores = {
            TaskType.COMPUTATION: 0,
            TaskType.DATA_ANALYSIS: 0,
            TaskType.REASONING: 0,
            TaskType.CREATIVE: 0,
            TaskType.CODE_GENERATION: 0,
            TaskType.CONVERSATION: 0,
        }

        # Check patterns
        for pattern in self.COMPUTATION_PATTERNS:
            if re.search(pattern, task, re.IGNORECASE):
                scores[TaskType.COMPUTATION] += 1

        for pattern in self.DATA_ANALYSIS_PATTERNS:
            if re.search(pattern, task, re.IGNORECASE):
                scores[TaskType.DATA_ANALYSIS] += 1

        for pattern in self.REASONING_PATTERNS:
            if re.search(pattern, task, re.IGNORECASE):
                scores[TaskType.REASONING] += 1

        for pattern in self.CREATIVE_PATTERNS:
            if re.search(pattern, task, re.IGNORECASE):
                scores[TaskType.CREATIVE] += 1

        for pattern in self.CODE_PATTERNS:
            if re.search(pattern, task, re.IGNORECASE):
                scores[TaskType.CODE_GENERATION] += 1

        # Find highest score
        max_score = max(scores.values())
        if max_score == 0:
            return TaskType.CONVERSATION

        for task_type, score in scores.items():
            if score == max_score:
                return task_type

        return TaskType.UNKNOWN

    def _extract_operations(self, task: str) -> List[str]:
        """Extract mathematical operations from task."""
        operations = []

        # Only match operations as whole words or symbols, not as substrings
        # e.g., "exp" should not match in "explain"
        for op in self.SYMBOLIC_OPERATIONS:
            # For single character operators, match directly
            if len(op) == 1 or op in ("+", "-", "*", "/", "^", "**", "%"):
                if op in task:
                    operations.append(op)
            else:
                # For word operators (sqrt, sin, cos, etc.), match as whole words
                # Use word boundaries to avoid matching substrings
                import re
                pattern = r'\b' + re.escape(op) + r'\b'
                if re.search(pattern, task, re.IGNORECASE):
                    operations.append(op)

        return operations

    def _extract_data_references(self, task: str) -> List[str]:
        """Extract references to data sources."""
        refs = []

        # File references
        file_patterns = [
            r"(\w+\.csv)",
            r"(\w+\.xlsx?)",
            r"(\w+\.json)",
            r"(\w+\.txt)",
        ]
        for pattern in file_patterns:
            matches = re.findall(pattern, task, re.IGNORECASE)
            refs.extend(matches)

        # Variable references
        var_pattern = r"(?:variable|data|dataset|dataframe)\s+['\"]?(\w+)['\"]?"
        matches = re.findall(var_pattern, task, re.IGNORECASE)
        refs.extend(matches)

        return refs

    def _estimate_complexity(
        self,
        task: str,
        numbers: List[str],
        operations: List[str],
        data_refs: List[str],
    ) -> float:
        """Estimate task complexity (0-1)."""
        score = 0.0

        # Length factor
        words = task.split()
        if len(words) > 50:
            score += 0.2
        elif len(words) > 20:
            score += 0.1

        # Numbers factor
        if len(numbers) > 5:
            score += 0.2
        elif len(numbers) > 2:
            score += 0.1

        # Operations factor
        if len(operations) > 3:
            score += 0.2
        elif len(operations) > 0:
            score += 0.1

        # Data factor
        if data_refs:
            score += 0.2

        # Multi-step indicator
        step_indicators = ["then", "after", "next", "finally", "first", "second"]
        if any(ind in task.lower() for ind in step_indicators):
            score += 0.2

        return min(score, 1.0)

    def _suggest_tools(
        self,
        task_type: TaskType,
        requires_precision: bool,
        requires_external_data: bool,
    ) -> List[str]:
        """Suggest tools for the task."""
        tools = []

        if requires_precision or task_type == TaskType.COMPUTATION:
            tools.append("execute_code")

        if task_type == TaskType.DATA_ANALYSIS:
            tools.append("execute_code")  # For pandas/numpy

        if requires_external_data:
            tools.append("read_file")

        if task_type in (TaskType.REASONING, TaskType.CREATIVE):
            tools.append("llm_call")

        return tools

    def assess_neuro_fit(self, characteristics: TaskCharacteristics) -> float:
        """
        Assess how well neural computation fits the task.

        Returns:
            Score from 0 (poor fit) to 1 (excellent fit)
        """
        score = 0.5  # Base score

        # Neural is good for
        if characteristics.task_type in (TaskType.REASONING, TaskType.CREATIVE, TaskType.CONVERSATION):
            score += 0.3

        if characteristics.requires_creativity:
            score += 0.2

        # Neural is less good for
        if characteristics.requires_precision:
            score -= 0.3

        if len(characteristics.numbers) > 3:
            score -= 0.2

        if len(characteristics.operations) > 0:
            score -= 0.2

        return max(0.0, min(1.0, score))

    def assess_symbolic_fit(self, characteristics: TaskCharacteristics) -> float:
        """
        Assess how well symbolic computation fits the task.

        Returns:
            Score from 0 (poor fit) to 1 (excellent fit)
        """
        score = 0.5  # Base score

        # Symbolic is good for
        if characteristics.task_type in (TaskType.COMPUTATION, TaskType.DATA_ANALYSIS):
            score += 0.3

        if characteristics.requires_precision:
            score += 0.2

        if len(characteristics.numbers) > 3:
            score += 0.2

        if len(characteristics.operations) > 0:
            score += 0.2

        if characteristics.requires_iteration:
            score += 0.1

        # Symbolic is less good for
        if characteristics.requires_creativity:
            score -= 0.3

        return max(0.0, min(1.0, score))

    # Patterns for semantic understanding detection
    SEMANTIC_UNDERSTANDING_PATTERNS = [
        # Topic/theme analysis
        r"(?:identify|find|detect|discover|extract)\s+(?:the\s+)?(?:main\s+)?(?:topics?|themes?|subjects?)",
        r"topic\s+(?:analysis|modeling|extraction|identification)",
        r"what\s+(?:are|is)\s+(?:the\s+)?(?:main\s+)?(?:topics?|themes?)",
        # Sentiment analysis
        r"(?:analyze|detect|determine|assess)\s+(?:the\s+)?sentiment",
        r"sentiment\s+(?:analysis|classification|detection)",
        r"(?:positive|negative|neutral)\s+(?:sentiment|feeling|opinion)",
        # Content classification
        r"(?:classify|categorize|label)\s+(?:the\s+)?(?:content|text|documents?|reviews?)",
        r"content\s+(?:classification|categorization)",
        # Intent recognition
        r"(?:identify|detect|recognize|understand)\s+(?:the\s+)?(?:intent|intention|purpose)",
        r"intent\s+(?:recognition|detection|classification)",
        # Summarization
        r"(?:summarize|summarise|condense|abstract)\s+(?:the\s+)?(?:content|text|documents?|articles?)",
        r"(?:create|generate|write)\s+(?:a\s+)?summary",
        # Entity extraction
        r"(?:extract|identify|find)\s+(?:named\s+)?entities",
        r"(?:named\s+)?entity\s+(?:recognition|extraction)",
        # Semantic similarity
        r"(?:semantic|meaning)\s+(?:similarity|comparison)",
        r"(?:similar|related)\s+(?:meaning|content)",
        # Understanding/interpretation
        r"(?:understand|interpret|comprehend)\s+(?:the\s+)?(?:meaning|context|nuance)",
    ]

    # Patterns for enumerable (closed-set) tasks
    ENUMERABLE_TASK_PATTERNS = [
        # Explicit categories
        r"classify\s+(?:as|into)\s+(?:one\s+of\s+)?(?:\d+\s+)?(?:categories|classes|groups|types)",
        r"(?:categorize|label)\s+(?:as|into)\s+(?:A|B|C|positive|negative|spam|ham)",
        r"(?:is\s+this|determine\s+if)\s+(?:spam|fraud|fake|valid|true|false)",
        # Binary classification
        r"(?:yes|no)\s+(?:or|/)\s+(?:yes|no)",
        r"(?:true|false)\s+(?:or|/)\s+(?:true|false)",
        r"(?:positive|negative)\s+(?:or|/)\s+(?:positive|negative)",
        # Predefined options
        r"choose\s+(?:from|between)\s+(?:the\s+)?(?:following|options)",
        r"select\s+(?:one|the\s+best)\s+(?:from|of)",
        r"which\s+(?:of\s+the\s+following|category|option)",
    ]

    # Patterns for open-ended (non-enumerable) tasks
    OPEN_ENDED_TASK_PATTERNS = [
        # Discovery tasks
        r"(?:discover|find|identify)\s+(?:all|any|the)\s+(?:topics?|themes?|patterns?|trends?)",
        r"what\s+(?:topics?|themes?|patterns?)\s+(?:are|exist|emerge)",
        # Exploration tasks
        r"(?:explore|investigate|examine)\s+(?:the\s+)?(?:content|data|text)",
        # Generation without constraints
        r"(?:generate|create|write)\s+(?:a\s+)?(?:list|summary)\s+of\s+(?:all|any)",
        # Analysis without predefined categories
        r"(?:analyze|analyse)\s+(?:the\s+)?(?:content|text|data)\s+(?:for|to\s+find)",
    ]

    def requires_semantic_understanding(self, task: str) -> bool:
        """
        Determine if a task requires semantic understanding.

        Tasks requiring semantic understanding should use Neural Computing
        because LLMs excel at meaning, context, and nuance.

        Args:
            task: Task description

        Returns:
            True if the task requires semantic understanding
        """
        task_lower = task.lower()

        # Check for semantic understanding patterns
        for pattern in self.SEMANTIC_UNDERSTANDING_PATTERNS:
            if re.search(pattern, task_lower, re.IGNORECASE):
                return True

        # Check for semantic keywords
        semantic_keywords = [
            "sentiment", "emotion", "feeling", "opinion",
            "topic", "theme", "subject", "meaning",
            "intent", "intention", "purpose",
            "summarize", "summarise", "abstract",
            "understand", "interpret", "comprehend",
            "context", "nuance", "implication",
            "tone", "style", "voice",
        ]

        return any(kw in task_lower for kw in semantic_keywords)

    def is_enumerable_task(self, task: str) -> bool:
        """
        Determine if a task can be solved through enumeration.

        Enumerable tasks have a finite, predefined set of possible outputs
        and can potentially use Symbolic Computing with hardcoded rules.

        Non-enumerable (open-ended) tasks have infinite possible outputs
        and should use Neural Computing for discovery.

        Args:
            task: Task description

        Returns:
            True if the task is enumerable (closed-set)
            False if the task is open-ended (requires discovery)
        """
        task_lower = task.lower()

        # Check for explicit enumerable patterns
        for pattern in self.ENUMERABLE_TASK_PATTERNS:
            if re.search(pattern, task_lower, re.IGNORECASE):
                return True

        # Check for open-ended patterns (these override enumerable)
        for pattern in self.OPEN_ENDED_TASK_PATTERNS:
            if re.search(pattern, task_lower, re.IGNORECASE):
                return False

        # Check for explicit category lists (e.g., "classify as A, B, or C")
        category_list_pattern = r"(?:classify|categorize|label)\s+(?:as|into)\s+([A-Za-z]+(?:\s*,\s*[A-Za-z]+)+(?:\s*(?:or|and)\s*[A-Za-z]+)?)"
        if re.search(category_list_pattern, task_lower, re.IGNORECASE):
            return True

        # Default: if task mentions "identify topics/themes" without predefined list, it's open-ended
        open_ended_keywords = [
            "identify topics", "find topics", "discover topics",
            "identify themes", "find themes", "discover themes",
            "identify patterns", "find patterns", "discover patterns",
            "what topics", "what themes", "what patterns",
        ]

        if any(kw in task_lower for kw in open_ended_keywords):
            return False

        # Default to enumerable for simple classification tasks
        if "classify" in task_lower or "categorize" in task_lower:
            # Check if categories are mentioned
            if any(word in task_lower for word in ["into", "as", "categories", "classes"]):
                return True

        # Default: assume non-enumerable (safer for semantic tasks)
        return False

    def get_semantic_task_recommendation(self, task: str) -> Dict[str, Any]:
        """
        Get recommendation for handling a semantic task.

        Args:
            task: Task description

        Returns:
            Dictionary with recommendation details
        """
        requires_semantic = self.requires_semantic_understanding(task)
        is_enumerable = self.is_enumerable_task(task)

        recommendation = {
            "requires_semantic_understanding": requires_semantic,
            "is_enumerable": is_enumerable,
            "recommended_mode": "neural",  # Default
            "reasoning": "",
        }

        if requires_semantic:
            if is_enumerable:
                recommendation["recommended_mode"] = "hybrid"
                recommendation["reasoning"] = (
                    "Task requires semantic understanding but has predefined categories. "
                    "Use Neural for understanding, but Symbolic can help with final classification."
                )
            else:
                recommendation["recommended_mode"] = "neural"
                recommendation["reasoning"] = (
                    "Task requires semantic understanding with open-ended discovery. "
                    "Neural Computing is essential - do NOT hardcode categories."
                )
        else:
            if is_enumerable:
                recommendation["recommended_mode"] = "symbolic"
                recommendation["reasoning"] = (
                    "Task has predefined categories and doesn't require deep semantic understanding. "
                    "Symbolic Computing with rules may be sufficient."
                )
            else:
                recommendation["recommended_mode"] = "neural"
                recommendation["reasoning"] = (
                    "Task is open-ended and may benefit from Neural Computing for discovery."
                )

        return recommendation
