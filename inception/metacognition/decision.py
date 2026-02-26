"""
Computation Mode Decision.

Decides whether to use neural or symbolic computation for a task.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from inception.metacognition.capability import CapabilityAssessor, TaskCharacteristics
from inception.metacognition.data_estimator import (
    DataScale,
    DataSizeEstimate,
    DataSizeEstimator,
    ProcessingStrategy,
)


class ComputationMode(str, Enum):
    """Computation modes."""
    NEURAL = "neural"  # LLM-based reasoning
    SYMBOLIC = "symbolic"  # Code execution
    HYBRID = "hybrid"  # Both
    TOOL = "tool"  # Use specific tool


@dataclass
class Decision:
    """A computation mode decision."""
    mode: ComputationMode
    confidence: float
    reasoning: str
    # For tool mode
    suggested_tool: Optional[str] = None
    # For hybrid mode
    steps: Optional[List[Dict[str, Any]]] = None


@dataclass
class EnhancedDecision(Decision):
    """
    Enhanced decision with data scale awareness and processing strategy.

    Extends the base Decision with additional information for handling
    large datasets and semantic tasks.
    """
    # Data scale information
    data_estimate: Optional[DataSizeEstimate] = None
    # Processing strategy
    processing_strategy: ProcessingStrategy = ProcessingStrategy.DIRECT
    # Recommended batch size (if batching needed)
    recommended_batch_size: int = 0
    # Whether task requires semantic understanding
    requires_semantic: bool = False
    # Whether task is enumerable (closed-set)
    is_enumerable: bool = True
    # Orchestration plan for complex tasks
    orchestration_plan: Optional[List[Dict[str, Any]]] = None
    # Warnings for the agent
    warnings: List[str] = field(default_factory=list)

    def to_guidance(self) -> str:
        """
        Convert decision to human-readable guidance for the agent.

        Returns:
            Formatted guidance string
        """
        lines = []

        # Header
        lines.append(f"**Recommended Mode**: {self.mode.value.upper()}")
        lines.append(f"**Confidence**: {self.confidence:.0%}")
        lines.append(f"**Reasoning**: {self.reasoning}")

        # Warnings
        if self.warnings:
            lines.append("")
            for warning in self.warnings:
                lines.append(f"**Warning**: {warning}")

        # Data scale information (without orchestration plan, we'll add it once at the end)
        if self.data_estimate and self.data_estimate.requires_batching():
            lines.append("")
            lines.append("---")
            lines.append("**Data Scale Assessment**:")
            lines.append(f"**Data Scale**: {self.data_estimate.scale.value.upper()}")
            lines.append(f"**Estimated Items**: ~{self.data_estimate.estimated_items:,}")
            lines.append(f"**Estimated Tokens**: ~{self.data_estimate.estimated_tokens:,}")
            lines.append(f"**Strategy**: {self.data_estimate.strategy.value}")
            lines.append(f"**Recommended Batch Size**: {self.data_estimate.recommended_batch_size}")

        # Semantic task information
        if self.requires_semantic:
            lines.append("")
            if not self.is_enumerable:
                lines.append("**Semantic Task**: Open-ended discovery required")
                lines.append("  - Do NOT hardcode topic/theme lists")
                lines.append("  - Use Neural Computing for semantic discovery")
            else:
                lines.append("**Semantic Task**: Closed-set classification")
                lines.append("  - Predefined categories detected")
                lines.append("  - Can use hybrid approach")

        # Orchestration plan (only once)
        if self.orchestration_plan:
            lines.append("")
            lines.append("**Orchestration Plan**:")
            for i, step in enumerate(self.orchestration_plan, 1):
                mode = step.get("mode", "unknown")
                purpose = step.get("purpose", "")
                lines.append(f"  {i}. [{mode}] {purpose}")

        return "\n".join(lines)


class ComputationDecider:
    """
    Decides the computation mode for a task.

    Uses task characteristics and historical performance to make decisions.
    """

    # Thresholds for mode selection
    NEURO_THRESHOLD = 0.6
    SYMBOLIC_THRESHOLD = 0.6
    HYBRID_THRESHOLD = 0.4  # When both are moderate

    def __init__(
        self,
        assessor: Optional[CapabilityAssessor] = None,
        data_estimator: Optional[DataSizeEstimator] = None,
    ):
        """
        Initialize the decider.

        Args:
            assessor: Capability assessor (creates new if not provided)
            data_estimator: Data size estimator (creates new if not provided)
        """
        self._assessor = assessor or CapabilityAssessor()
        self._data_estimator = data_estimator or DataSizeEstimator()
        self._decision_history: List[Dict[str, Any]] = []

    def decide(self, task: str, execution_context: Optional[Dict[str, Any]] = None) -> EnhancedDecision:
        """
        Decide the computation mode for a task.

        Args:
            task: Task description
            execution_context: Optional context with variable information

        Returns:
            EnhancedDecision with mode, strategy, and reasoning
        """
        # Assess task characteristics
        characteristics = self._assessor.assess(task)

        # Estimate data size
        data_estimate = self._data_estimator.estimate(task, execution_context)

        # Check semantic understanding requirements
        requires_semantic = self._assessor.requires_semantic_understanding(task)
        is_enumerable = self._assessor.is_enumerable_task(task)

        # Get fit scores
        neuro_fit = self._assessor.assess_neuro_fit(characteristics)
        symbolic_fit = self._assessor.assess_symbolic_fit(characteristics)

        # Adjust scores based on semantic requirements
        if requires_semantic and not is_enumerable:
            # Open-ended semantic tasks strongly favor neural
            neuro_fit = max(neuro_fit, 0.8)
            symbolic_fit = min(symbolic_fit, 0.3)

        # Adjust based on data scale
        if data_estimate.requires_batching():
            # Large data requires hybrid approach
            # Can't be pure neural (token overflow) or pure symbolic (needs semantic)
            if requires_semantic:
                # Force hybrid with map-reduce
                neuro_fit = 0.5
                symbolic_fit = 0.5

        # Decide base mode
        if neuro_fit > self.NEURO_THRESHOLD and symbolic_fit < self.HYBRID_THRESHOLD:
            base_decision = self._decide_neural(characteristics, neuro_fit)
        elif symbolic_fit > self.SYMBOLIC_THRESHOLD and neuro_fit < self.HYBRID_THRESHOLD:
            base_decision = self._decide_symbolic(characteristics, symbolic_fit)
        elif characteristics.suggested_tools:
            base_decision = self._decide_tool(characteristics)
        else:
            base_decision = self._decide_hybrid(characteristics, neuro_fit, symbolic_fit)

        # Build enhanced decision
        warnings = []

        # Add warnings for medium/large data
        if data_estimate.scale == DataScale.LARGE:
            warnings.append(
                f"Large dataset detected (~{data_estimate.estimated_items:,} items). "
                "MUST use MapReduce processing."
            )
        elif data_estimate.scale == DataScale.MEDIUM:
            warnings.append(
                f"Medium dataset detected (~{data_estimate.estimated_items:,} items). "
                "Should use batch processing."
            )

        # Add warnings for semantic tasks with hardcoding risk
        if requires_semantic and not is_enumerable:
            warnings.append(
                "Open-ended semantic task detected. "
                "Do NOT hardcode topic/category lists - use Neural for discovery."
            )

        # Override mode for large semantic tasks
        final_mode = base_decision.mode
        if data_estimate.requires_batching() and requires_semantic:
            final_mode = ComputationMode.HYBRID

        # Build orchestration plan
        orchestration_plan = self._build_orchestration_plan(
            task, final_mode, data_estimate, requires_semantic
        )

        # Build reasoning
        reasoning_parts = [base_decision.reasoning]
        if data_estimate.requires_batching():
            reasoning_parts.append(
                f"Data scale ({data_estimate.scale.value}) requires {data_estimate.strategy.value} processing"
            )
        if requires_semantic:
            semantic_type = "open-ended" if not is_enumerable else "closed-set"
            reasoning_parts.append(f"Semantic understanding required ({semantic_type})")

        enhanced_decision = EnhancedDecision(
            mode=final_mode,
            confidence=base_decision.confidence,
            reasoning="; ".join(reasoning_parts),
            suggested_tool=base_decision.suggested_tool,
            steps=base_decision.steps,
            data_estimate=data_estimate,
            processing_strategy=data_estimate.strategy,
            recommended_batch_size=data_estimate.recommended_batch_size,
            requires_semantic=requires_semantic,
            is_enumerable=is_enumerable,
            orchestration_plan=orchestration_plan,
            warnings=warnings,
        )

        # Record decision
        self._decision_history.append({
            "task": task,
            "characteristics": characteristics,
            "data_estimate": data_estimate,
            "neuro_fit": neuro_fit,
            "symbolic_fit": symbolic_fit,
            "decision": enhanced_decision,
        })

        return enhanced_decision

    def _build_orchestration_plan(
        self,
        task: str,
        mode: ComputationMode,
        data_estimate: DataSizeEstimate,
        requires_semantic: bool,
    ) -> Optional[List[Dict[str, Any]]]:
        """Build orchestration plan for complex tasks."""
        # Use data estimator's plan if available (for batch/map-reduce)
        if data_estimate.orchestration_plan:
            return data_estimate.orchestration_plan

        # Build plan for hybrid mode without batching
        if mode == ComputationMode.HYBRID and not data_estimate.requires_batching():
            return [
                {"mode": "Neural", "purpose": "Analyze task and plan approach"},
                {"mode": "Symbolic", "purpose": "Execute computation/data processing"},
                {"mode": "Neural", "purpose": "Interpret and present results"},
            ]

        return None

    def _decide_neural(
        self,
        characteristics: TaskCharacteristics,
        confidence: float,
    ) -> Decision:
        """Create a neural mode decision."""
        reasons = []

        if characteristics.requires_creativity:
            reasons.append("Task requires creative generation")
        if characteristics.task_type.value in ("reasoning", "conversation", "creative"):
            reasons.append(f"Task type '{characteristics.task_type.value}' suits neural computation")
        if not characteristics.requires_precision:
            reasons.append("Precision is not critical")

        return Decision(
            mode=ComputationMode.NEURAL,
            confidence=confidence,
            reasoning="; ".join(reasons) if reasons else "Neural computation is most suitable",
        )

    def _decide_symbolic(
        self,
        characteristics: TaskCharacteristics,
        confidence: float,
    ) -> Decision:
        """Create a symbolic mode decision."""
        reasons = []

        if characteristics.requires_precision:
            reasons.append("Task requires precise computation")
        if characteristics.numbers:
            reasons.append(f"Task involves numbers: {characteristics.numbers[:3]}")
        if characteristics.operations:
            reasons.append(f"Task involves operations: {characteristics.operations}")
        if characteristics.requires_iteration:
            reasons.append("Task requires iteration")

        return Decision(
            mode=ComputationMode.SYMBOLIC,
            confidence=confidence,
            reasoning="; ".join(reasons) if reasons else "Symbolic computation is most suitable",
        )

    def _decide_tool(self, characteristics: TaskCharacteristics) -> Decision:
        """Create a tool mode decision."""
        suggested = characteristics.suggested_tools[0] if characteristics.suggested_tools else None

        return Decision(
            mode=ComputationMode.TOOL,
            confidence=0.7,
            reasoning=f"Specific tool recommended: {suggested}",
            suggested_tool=suggested,
        )

    def _decide_hybrid(
        self,
        characteristics: TaskCharacteristics,
        neuro_fit: float,
        symbolic_fit: float,
    ) -> Decision:
        """Create a hybrid mode decision."""
        # Plan hybrid execution
        steps = []

        # Usually: neural for planning, symbolic for execution
        steps.append({
            "mode": ComputationMode.NEURAL,
            "purpose": "Analyze and plan approach",
        })

        if characteristics.requires_precision or characteristics.operations:
            steps.append({
                "mode": ComputationMode.SYMBOLIC,
                "purpose": "Execute precise computation",
            })

        steps.append({
            "mode": ComputationMode.NEURAL,
            "purpose": "Interpret and present results",
        })

        return Decision(
            mode=ComputationMode.HYBRID,
            confidence=(neuro_fit + symbolic_fit) / 2,
            reasoning="Task benefits from both neural and symbolic computation",
            steps=steps,
        )

    def get_history(self) -> List[Dict[str, Any]]:
        """Get decision history."""
        return self._decision_history.copy()

    def learn_from_outcome(
        self,
        task: str,
        decision: Decision,
        success: bool,
        feedback: Optional[str] = None,
    ) -> None:
        """
        Learn from a decision outcome.

        Args:
            task: The task that was executed
            decision: The decision that was made
            success: Whether the execution was successful
            feedback: Optional feedback about the outcome
        """
        # Find the decision in history
        for entry in self._decision_history:
            if entry["task"] == task:
                entry["outcome"] = {
                    "success": success,
                    "feedback": feedback,
                }
                break

        # In a more sophisticated implementation, this would:
        # - Adjust thresholds based on success rates
        # - Learn task patterns that work well with each mode
        # - Build a model to predict the best mode
