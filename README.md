# Nestlet

A lightweight neuro-symbolic AI agent with recursive reasoning and autonomous tool-making.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![中文](https://img.shields.io/badge/lang-中文-red.svg)](README.zh.md)

## What Makes Nestlet Different

**Recursive Neural-Symbolic Architecture**: Unlike traditional agents where neural (LLM) calls symbolic (tools) in a flat hierarchy, Nestlet supports infinite nesting — symbolic tools can invoke neural reasoning, which can call more symbolic tools, and so on.

```
Traditional Agent:          Nestlet:

Neural ──► Symbolic         Neural ──► Symbolic ──► Neural ──► Symbolic ──► ...
   │                           │                        │
   └── done                    └── can recurse          └── can recurse
```

**Autonomous Tool Creation (LATM)**: When existing tools are insufficient, Nestlet creates new ones on the fly using the LLMs-as-Tool-Makers paradigm.

## Quick Start

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/jxtse/nestlet.git
cd nestlet
uv sync

# Configure
cp config.example.yaml config.yaml
# Edit config.yaml with your settings

# Set your API key
export OPENAI_API_KEY="your-api-key"

# Run
uv run python main.py
```

## Usage

### Interactive Mode

```bash
uv run python main.py
```

### Single Task

```bash
uv run python main.py --task "Calculate the factorial of 100"
```

### As a Library

```python
import asyncio
from nestlet import HybridAgent, Settings

async def main():
    agent = HybridAgent(settings=Settings.from_env())
    response = await agent.chat("Analyze this data and create a visualization tool for it")
    print(response)

asyncio.run(main())
```

## Core Concepts

| Concept | Description |
|---------|-------------|
| **Think-Act Loop** | LLM reasons → decides action → executes → updates context → repeats |
| **Nested Reasoning** | Tools can invoke LLM calls, enabling multi-level problem decomposition |
| **Tool Factory** | Dynamically creates, validates, and registers new tools at runtime |
| **Stateful Kernel** | Python execution environment with persistent variables across calls |

## Architecture Overview

```
inception/
├── agent/          # Think-Act loop, context management
├── tool/           # Tool registry, factory, built-in tools
├── executor/       # Sandboxed Python kernel
├── provider/       # LLM abstraction (OpenAI, Anthropic, Azure)
├── memory/         # Conversation + working memory
└── metacognition/  # Task analysis, computation mode selection
```

→ See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed module documentation.

## Configuration

Copy the example configuration and customize:

```bash
cp config.example.yaml config.yaml
```

```yaml
provider:
  type: openai          # openai | anthropic | azure | openrouter
  model: gpt-5.2
  temperature: 0.7

execution:
  mode: sandbox         # sandbox | trusted
  timeout: 30.0
```

Or use environment variables:

```bash
export INCEPTION_PROVIDER=openai
export INCEPTION_MODEL=gpt-5.2
export OPENAI_API_KEY=sk-...
```

## Supported Providers

- OpenAI (GPT-4, GPT-5, etc.)
- Anthropic (Claude 3.5, Claude 4)
- OpenRouter (access 100+ models)
- Azure OpenAI
- Any OpenAI-compatible API

## Development

```bash
# Install dev dependencies
uv sync --extra dev

# Run tests
uv run pytest tests/

# Format & lint
uv run black . && uv run ruff check .
```

## Acknowledgments

Inspired by:
- [Claude Code](https://claude.ai/code) / [OpenCode](https://github.com/opencode-ai/opencode) — CLI agent patterns
- [TaskWeaver](https://github.com/microsoft/TaskWeaver) — Stateful execution
- [LATM](https://arxiv.org/abs/2305.17126) — LLMs as Tool Makers paradigm

## License

MIT
