"""
Inception - Neuro-Symbolic General Agent

Main entry point for the agent.
"""

import asyncio
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from inception import HybridAgent, Settings


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

console = Console()


async def interactive_session(agent: HybridAgent) -> None:
    """Run an interactive chat session with the agent."""
    console.print(Panel.fit(
        "[bold blue]Inception[/bold blue] - Neuro-Symbolic Agent\n"
        "Type 'exit' or 'quit' to end the session.\n"
        "Type 'reset' to clear the conversation.",
        title="Welcome",
    ))

    while True:
        try:
            # Get user input
            user_input = Prompt.ask("\n[bold green]You[/bold green]")

            if not user_input.strip():
                continue

            # Handle special commands
            if user_input.lower() in ("exit", "quit"):
                console.print("\n[bold]Goodbye![/bold]")
                break

            if user_input.lower() == "reset":
                agent.reset()
                console.print("[yellow]Conversation reset.[/yellow]")
                continue

            if user_input.lower() == "help":
                console.print(Panel(
                    "**Commands:**\n"
                    "- `exit` / `quit` - End the session\n"
                    "- `reset` - Clear conversation history\n"
                    "- `help` - Show this help message\n\n"
                    "**Features:**\n"
                    "- Ask questions and get answers\n"
                    "- Request calculations (I'll execute code)\n"
                    "- Ask for data analysis\n"
                    "- Create new tools for recurring tasks\n"
                    "- Send images: `[image:path/to/image.png] your question`\n"
                    "- Parse Office files: Ask to parse Word/Excel/PowerPoint/PDF files",
                    title="Help",
                ))
                continue

            # Parse images from input (format: [image:path] or [img:path])
            images = []
            import re
            image_pattern = r'\[(?:image|img):([^\]]+)\]'
            image_matches = re.findall(image_pattern, user_input)
            if image_matches:
                images = [m.strip() for m in image_matches]
                # Remove image tags from input
                user_input = re.sub(image_pattern, '', user_input).strip()
                if images:
                    console.print(f"[dim]📷 Attached {len(images)} image(s)[/dim]")

            # Get response from agent
            console.print("\n[bold blue]Inception[/bold blue]: ", end="")

            with console.status("[bold yellow]Thinking...[/bold yellow]"):
                response = await agent.chat(user_input, images=images if images else None)

            # Display response
            if "```" in response:
                # Response contains code blocks, use markdown
                console.print(Markdown(response))
            else:
                console.print(response)

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]")
            continue
        except Exception as e:
            logger.exception("Error during chat")
            console.print(f"\n[red]Error: {e}[/red]")


async def run_task(agent: HybridAgent, task: str) -> None:
    """Run a single task and display the result."""
    console.print(Panel.fit(f"[bold]Task:[/bold] {task}", title="Running Task"))

    with console.status("[bold yellow]Processing...[/bold yellow]"):
        response = await agent.run(task)

    console.print(Panel(Markdown(response), title="Result"))


def main() -> int:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Inception - Neuro-Symbolic General Agent"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration file (YAML)",
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Run a single task instead of interactive mode",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Configure logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)

    # Load settings
    if args.config and args.config.exists():
        settings = Settings.from_yaml(args.config)
    else:
        settings = Settings.from_env()

    settings.verbose = args.verbose
    settings.debug = args.debug

    # Create agent
    agent = HybridAgent(settings=settings)

    # Run
    try:
        if args.task:
            asyncio.run(run_task(agent, args.task))
        else:
            asyncio.run(interactive_session(agent))
        return 0
    except KeyboardInterrupt:
        console.print("\n[bold]Interrupted.[/bold]")
        return 130
    except Exception as e:
        logger.exception("Fatal error")
        console.print(f"\n[red bold]Fatal error:[/red bold] {e}")
        return 1
    finally:
        # Ensure tools are saved on exit
        agent.shutdown()


if __name__ == "__main__":
    sys.exit(main())
