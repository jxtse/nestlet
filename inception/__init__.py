"""
Inception - Neuro-Symbolic General Agent

A hybrid agent framework combining neural computation (LLM reasoning)
with symbolic computation (code execution) and autonomous tool-making capabilities.
"""

__version__ = "0.1.0"
__author__ = "Inception Team"

from inception.agent.hybrid_agent import HybridAgent
from inception.config.settings import Settings

__all__ = ["HybridAgent", "Settings", "__version__"]
