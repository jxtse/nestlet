"""
Example: Text Analysis with Inception

Demonstrates using the hybrid agent for text analysis tasks.
"""

import asyncio
import logging
from inception import HybridAgent, Settings


logging.basicConfig(level=logging.INFO)


async def main():
    """Run text analysis examples."""
    # Initialize agent
    settings = Settings.from_env()
    agent = HybridAgent(settings=settings)

    print("=" * 60)
    print("Inception - Text Analysis Example")
    print("=" * 60)

    # Example 1: Simple calculation
    print("\n--- Example 1: Calculation ---")
    response = await agent.chat(
        "Calculate the average of these numbers: 15, 23, 42, 8, 31"
    )
    print(f"Response: {response}")

    # Example 2: Text analysis
    print("\n--- Example 2: Text Analysis ---")
    sample_text = """
    Machine learning is a subset of artificial intelligence that enables
    systems to learn and improve from experience without being explicitly
    programmed. It focuses on developing algorithms that can access data
    and use it to learn for themselves.
    """
    response = await agent.chat(
        f"Analyze this text and extract the key concepts:\n\n{sample_text}"
    )
    print(f"Response: {response}")

    # Example 3: Data processing
    print("\n--- Example 3: Data Processing ---")
    response = await agent.chat(
        """Create a frequency distribution of the following words and show the top 5:
        apple banana apple cherry banana apple date banana cherry apple"""
    )
    print(f"Response: {response}")

    # Example 4: Multi-step task
    print("\n--- Example 4: Multi-step Analysis ---")
    response = await agent.chat(
        """Analyze this sales data:
        Q1: 15000, Q2: 22000, Q3: 18000, Q4: 25000

        1. Calculate the total annual sales
        2. Find the average quarterly sales
        3. Identify the quarter with highest growth
        4. Calculate the growth percentage from Q1 to Q4
        """
    )
    print(f"Response: {response}")

    # Reset for clean state
    agent.reset()
    print("\n--- Agent state reset ---")

    # Example 5: Tool creation (advanced)
    print("\n--- Example 5: Custom Analysis ---")
    response = await agent.chat(
        """I have a list of temperatures in Celsius: [20, 25, 30, 15, 22, 28]
        Convert each to Fahrenheit and calculate the average in both scales."""
    )
    print(f"Response: {response}")


if __name__ == "__main__":
    asyncio.run(main())
