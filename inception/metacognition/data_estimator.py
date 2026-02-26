"""
Data Size Estimation.

Estimates data scale from task descriptions and recommends processing strategies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class DataScale(str, Enum):
    """Data scale categories based on item count."""
    SMALL = "small"       # < 200 items - Direct processing
    MEDIUM = "medium"     # 200-2000 items - Batch processing
    LARGE = "large"       # > 2000 items - MapReduce processing


class ProcessingStrategy(str, Enum):
    """Recommended processing strategies for different data scales."""
    DIRECT = "direct"        # Process all at once (small scale)
    BATCH = "batch"          # Split into batches and process (medium scale)
    MAP_REDUCE = "map_reduce"  # Map-Reduce pattern (large scale)


@dataclass
class DataSizeEstimate:
    """Estimation of data size and recommended processing approach."""
    scale: DataScale
    strategy: ProcessingStrategy
    estimated_items: int
    estimated_tokens: int
    recommended_batch_size: int
    reasoning: str
    # For batch/map-reduce strategy
    orchestration_plan: Optional[List[Dict[str, Any]]] = None

    def requires_batching(self) -> bool:
        """Check if the data scale requires batch processing."""
        return self.scale in (DataScale.MEDIUM, DataScale.LARGE)

    def to_guidance(self) -> str:
        """Convert estimate to human-readable guidance."""
        lines = []

        if self.requires_batching():
            lines.append(f"**Data Scale**: {self.scale.value.upper()}")
            lines.append(f"**Estimated Items**: ~{self.estimated_items:,}")
            lines.append(f"**Strategy**: {self.strategy.value}")
            if self.strategy == ProcessingStrategy.BATCH:
                lines.append(f"**Recommended Batch Size**: {self.recommended_batch_size}")
        else:
            lines.append(f"**Data Scale**: {self.scale.value} (direct processing OK)")

        if self.orchestration_plan:
            lines.append("\n**Processing Approaches**:")
            for i, step in enumerate(self.orchestration_plan, 1):
                mode = step.get("mode", "unknown")
                purpose = step.get("purpose", "")
                lines.append(f"  {i}. [{mode}] {purpose}")

        return "\n".join(lines)


class DataSizeEstimator:
    """
    Estimates data scale from task descriptions.

    Helps decide whether to use direct processing or batch processing,
    and recommends appropriate strategies for large datasets.

    Scale thresholds:
    - SMALL: < 200 items → Direct processing
    - MEDIUM: 200-2000 items → Batch processing
    - LARGE: > 2000 items → MapReduce processing
    """

    # Thresholds for data scale classification
    SMALL_THRESHOLD = 200      # Below this: direct processing
    LARGE_THRESHOLD = 2000     # Above this: map-reduce

    # Default batch size for medium scale
    DEFAULT_BATCH_SIZE = 50

    # Patterns for detecting data size indicators
    NUMBER_PATTERNS = [
        # Explicit counts: "5000 reviews", "10,000 items"
        (r"(\d{1,3}(?:,\d{3})*|\d+)\s*(?:rows?|items?|records?|entries?|lines?|reviews?|comments?|documents?|files?|emails?|messages?|transactions?|orders?|customers?|users?|products?)", "items"),
        # File sizes: "500MB file", "2GB dataset"
        (r"(\d+(?:\.\d+)?)\s*(?:KB|MB|GB|TB)", "file_size"),
        # Token counts: "100K tokens"
        (r"(\d+(?:\.\d+)?)\s*[Kk]\s*tokens?", "tokens"),
        # Large number indicators: "thousands of", "millions of"
        (r"(thousands?|millions?|billions?)\s+of\s+\w+", "magnitude"),
        # CSV/Excel indicators
        (r"(?:csv|excel|xlsx?)\s+(?:file|with)\s+(\d{1,3}(?:,\d{3})*|\d+)", "file_items"),
    ]

    # Keywords indicating large data operations
    LARGE_DATA_KEYWORDS = [
        "all", "every", "each", "entire", "whole", "complete",
        "batch", "bulk", "mass", "large", "huge", "massive",
        "dataset", "database", "corpus", "collection",
    ]

    # Keywords indicating semantic analysis tasks
    SEMANTIC_TASK_KEYWORDS = [
        "analyze", "analysis", "sentiment", "topic", "theme",
        "classify", "categorize", "summarize", "extract",
        "identify", "detect", "recognize", "understand",
    ]

    def __init__(self, default_tokens_per_item: int = 50):
        """
        Initialize the estimator.

        Args:
            default_tokens_per_item: Default token estimate per data item
        """
        self._default_tokens_per_item = default_tokens_per_item

    def estimate(self, task: str, execution_context: Optional[Dict[str, Any]] = None) -> DataSizeEstimate:
        """
        Estimate data size from task description.

        Args:
            task: Task description
            execution_context: Optional context with variable information

        Returns:
            DataSizeEstimate with scale and strategy recommendations
        """
        task_lower = task.lower()

        # Extract size indicators from task
        estimated_items, items_source = self._extract_item_count(task)

        # Check execution context for variable sizes
        if execution_context and "variables" in execution_context:
            context_items = self._estimate_from_context(execution_context["variables"])
            if context_items > estimated_items:
                estimated_items = context_items
                items_source = "context"

        # Estimate tokens
        estimated_tokens = self._estimate_tokens(estimated_items, task_lower)

        # Determine scale
        scale = self._classify_scale(estimated_items)

        # Determine strategy based on scale
        strategy = self._determine_strategy(scale)

        # Calculate batch size
        batch_size = self._calculate_batch_size(scale)

        # Build reasoning
        reasoning = self._build_reasoning(
            estimated_items, estimated_tokens, items_source, scale, strategy
        )

        # Build orchestration plan for batch/map-reduce strategies
        orchestration_plan = None
        if strategy in (ProcessingStrategy.BATCH, ProcessingStrategy.MAP_REDUCE):
            orchestration_plan = self._build_orchestration_plan(strategy, batch_size)

        return DataSizeEstimate(
            scale=scale,
            strategy=strategy,
            estimated_items=estimated_items,
            estimated_tokens=estimated_tokens,
            recommended_batch_size=batch_size,
            reasoning=reasoning,
            orchestration_plan=orchestration_plan,
        )

    def _extract_item_count(self, task: str) -> Tuple[int, str]:
        """Extract item count from task description."""
        max_items = 0
        source = "default"

        for pattern, pattern_type in self.NUMBER_PATTERNS:
            matches = re.findall(pattern, task, re.IGNORECASE)
            for match in matches:
                if pattern_type == "items" or pattern_type == "file_items":
                    # Parse number with commas
                    if isinstance(match, str):
                        num_str = match.replace(",", "")
                        try:
                            count = int(num_str)
                            if count > max_items:
                                max_items = count
                                source = "explicit"
                        except ValueError:
                            pass

                elif pattern_type == "file_size":
                    # Estimate items from file size (rough: 100 bytes per item)
                    try:
                        size = float(match)
                        # Assume KB/MB/GB based on pattern match
                        if "GB" in task.upper():
                            size *= 1024 * 1024 * 1024
                        elif "MB" in task.upper():
                            size *= 1024 * 1024
                        elif "KB" in task.upper():
                            size *= 1024
                        estimated = int(size / 100)  # ~100 bytes per item
                        if estimated > max_items:
                            max_items = estimated
                            source = "file_size"
                    except ValueError:
                        pass

                elif pattern_type == "tokens":
                    # Direct token count
                    try:
                        tokens = int(float(match) * 1000)  # K = 1000
                        estimated = tokens // self._default_tokens_per_item
                        if estimated > max_items:
                            max_items = estimated
                            source = "tokens"
                    except ValueError:
                        pass

                elif pattern_type == "magnitude":
                    # Magnitude indicators
                    if "million" in match.lower():
                        max_items = max(max_items, 1000000)
                        source = "magnitude"
                    elif "thousand" in match.lower():
                        max_items = max(max_items, 1000)
                        source = "magnitude"
                    elif "billion" in match.lower():
                        max_items = max(max_items, 1000000000)
                        source = "magnitude"

        # If no explicit count found, check for large data keywords
        if max_items == 0:
            task_lower = task.lower()
            if any(kw in task_lower for kw in self.LARGE_DATA_KEYWORDS):
                max_items = 100  # Conservative estimate
                source = "keywords"
            else:
                max_items = 10  # Default for simple tasks
                source = "default"

        return max_items, source

    def _estimate_from_context(self, variables: Dict[str, Any]) -> int:
        """Estimate item count from execution context variables."""
        max_items = 0

        for var_name, var_info in variables.items():
            if isinstance(var_info, dict):
                # Check for size/length info
                if "length" in var_info:
                    max_items = max(max_items, var_info["length"])
                elif "rows" in var_info:
                    max_items = max(max_items, var_info["rows"])
                elif "size" in var_info:
                    max_items = max(max_items, var_info["size"])
            elif isinstance(var_info, (list, tuple)):
                max_items = max(max_items, len(var_info))
            elif isinstance(var_info, str) and var_info.isdigit():
                max_items = max(max_items, int(var_info))

        return max_items

    def _estimate_tokens(self, items: int, task: str) -> int:
        """Estimate total tokens based on item count and task type."""
        tokens_per_item = self._default_tokens_per_item

        # Adjust based on task type
        if "review" in task or "comment" in task or "feedback" in task:
            tokens_per_item = 100  # Reviews tend to be longer
        elif "tweet" in task or "message" in task:
            tokens_per_item = 30  # Short messages
        elif "document" in task or "article" in task:
            tokens_per_item = 500  # Documents are longer
        elif "email" in task:
            tokens_per_item = 150  # Emails vary

        return items * tokens_per_item

    def _classify_scale(self, items: int) -> DataScale:
        """
        Classify data scale based on item count.

        - SMALL: < 200 items → Direct processing
        - MEDIUM: 200-2000 items → Batch processing
        - LARGE: > 2000 items → MapReduce processing
        """
        if items < self.SMALL_THRESHOLD:
            return DataScale.SMALL
        elif items <= self.LARGE_THRESHOLD:
            return DataScale.MEDIUM
        else:
            return DataScale.LARGE

    def _determine_strategy(self, scale: DataScale) -> ProcessingStrategy:
        """Determine processing strategy based on scale."""
        if scale == DataScale.SMALL:
            return ProcessingStrategy.DIRECT
        elif scale == DataScale.MEDIUM:
            return ProcessingStrategy.BATCH
        else:  # LARGE
            return ProcessingStrategy.MAP_REDUCE

    def _calculate_batch_size(self, scale: DataScale) -> int:
        """Calculate recommended batch size."""
        # For medium scale, use default batch size
        # For large scale (map-reduce), also use batch size for the map phase
        if scale in (DataScale.MEDIUM, DataScale.LARGE):
            return self.DEFAULT_BATCH_SIZE
        return 0  # No batching needed for small scale

    def _build_reasoning(
        self,
        items: int,
        tokens: int,
        source: str,
        scale: DataScale,
        strategy: ProcessingStrategy,
    ) -> str:
        """Build reasoning explanation."""
        parts = []

        parts.append(f"Detected {items:,} items (source: {source})")
        parts.append(f"Scale: {scale.value}")
        parts.append(f"Strategy: {strategy.value}")

        if scale == DataScale.LARGE:
            parts.append("Large dataset requires MapReduce processing")
        elif scale == DataScale.MEDIUM:
            parts.append("Medium dataset requires batch processing")

        return "; ".join(parts)

    def _build_orchestration_plan(
        self,
        strategy: ProcessingStrategy,
        batch_size: int,
    ) -> List[Dict[str, Any]]:
        """
        Build orchestration plan for batch/map-reduce processing.

        For BATCH strategy, provides two approaches for the agent to choose from:
        1. Batch approach: Split data into batches, process each batch with LLM
        2. Loop approach: Iterate one by one, call LLM for each item

        For MAP_REDUCE strategy, provides the standard map-reduce pattern.
        """
        plan = []

        if strategy == ProcessingStrategy.BATCH:
            # Two approaches for medium-scale batch processing
            plan = [
                {"mode": "Approach A", "purpose": "Batch Processing"},
                {"mode": "Symbolic", "purpose": f"  Split data into batches of ~{batch_size} items"},
                {"mode": "Loop", "purpose": "  For each batch:"},
                {"mode": "Neural", "purpose": "    Call llm_call to process the batch"},
                {"mode": "Symbolic", "purpose": "  Combine all batch results"},
                {"mode": "---", "purpose": ""},
                {"mode": "Approach B", "purpose": "Item-by-Item Processing"},
                {"mode": "Symbolic", "purpose": "  Write a for loop to iterate through items"},
                {"mode": "Loop", "purpose": "  For each item:"},
                {"mode": "Neural", "purpose": "    Call llm_call to process the single item"},
                {"mode": "Symbolic", "purpose": "  Collect and aggregate results"},
            ]

        elif strategy == ProcessingStrategy.MAP_REDUCE:
            plan = [
                {"mode": "Map Phase", "purpose": ""},
                {"mode": "Symbolic", "purpose": f"  Split data into batches of ~{batch_size} items"},
                {"mode": "Loop", "purpose": "  For each batch:"},
                {"mode": "Neural", "purpose": "    Call llm_call to analyze the batch"},
                {"mode": "Symbolic", "purpose": "    Store intermediate results"},
                {"mode": "Reduce Phase", "purpose": ""},
                {"mode": "Symbolic", "purpose": "  Aggregate all intermediate results"},
                {"mode": "Neural", "purpose": "  Call llm_call to synthesize final insights"},
            ]

        return plan
