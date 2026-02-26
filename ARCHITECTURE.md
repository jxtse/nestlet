# Inception

**Neuro-Symbolic General Agent** - A hybrid AI agent framework combining neural computation (LLM reasoning) with symbolic computation (code execution) and autonomous tool-making capabilities.

## Overview

Inception implements the LATM (LLMs as Tool Makers) paradigm, enabling an AI agent to:

1. **Neural Reasoning**: Use LLM capabilities for understanding, planning, and creative tasks
2. **Symbolic Computation**: Execute Python code for precise calculations and data processing
3. **Autonomous Tool Creation**: Create new tools when existing ones are insufficient

## Features

- 🧠 **Hybrid Intelligence**: Combines LLM reasoning with code execution
- 🛠️ **Tool System**: Extensible tool framework with built-in and generated tools
- 📝 **Stateful Execution**: Python kernel maintains state across executions
- 🔒 **Safe Execution**: Sandboxed code execution with security controls
- 💾 **Memory Management**: Conversation history and working memory
- 🎯 **Metacognition**: Self-assessment of capabilities and decision-making

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd inception

# Install with pip
pip install -e .

# Or with optional dependencies
pip install -e ".[dev]"  # Development dependencies
pip install -e ".[azure]"  # Azure OpenAI support
```

## Quick Start

### Interactive Mode

```bash
python main.py
```

### Run a Single Task

```bash
python main.py --task "Calculate the factorial of 100"
```

### Configuration File

Create a `config.yaml`:

```yaml
provider:
  type: openai
  model: gpt-5.2
  temperature: 0.7

execution:
  mode: sandbox
  timeout: 30.0

memory:
  max_conversation_turns: 50
```

Then run:

```bash
python main.py --config config.yaml
```

---

## Architecture

### Project Structure

```
Inception/
├── inception/               # Core framework code
│   ├── __init__.py         # Package entry, exports HybridAgent and Settings
│   ├── agent/              # Agent system
│   ├── config/             # Configuration management
│   ├── executor/           # Code execution engine
│   ├── memory/             # Memory system
│   ├── metacognition/      # Metacognition system
│   ├── planner/            # Task planning system
│   ├── provider/           # LLM provider abstraction
│   └── tool/               # Tool system
├── tests/                   # Test files
├── plugins/                 # Plugin directory (for dynamically generated tools)
├── examples/                # Example code
├── main.py                  # Main entry point
├── config.yaml              # Configuration file
└── pyproject.toml           # Project dependencies
```

### Module Details

#### 1. agent/ - Agent System

| File | Description |
|------|-------------|
| `base.py` | Defines agent base classes and core data structures (Context, Action, ActionType, etc.) |
| `hybrid_agent.py` | **Main hybrid agent implementation**, contains the Think-Act loop |

**Core Classes:**
- `BaseAgent` - Abstract base class for agents
- `HybridAgent` - Main agent implementation with Think-Act loop
- `Context` - Agent reasoning context (user input, conversation history, available tools)
- `Action` - Represents an action to be executed
- `ActionType` - Action type enum (TOOL_CALL, CODE_EXEC, LLM_CALL, CREATE_TOOL, RESPOND, DELEGATE, WAIT)
- `ThinkResult` - Thinking result with reasoning chain and decided action
- `ActionResult` - Action execution result

#### 2. tool/ - Tool System

| File | Description |
|------|-------------|
| `base.py` | Tool base classes and specification definitions |
| `registry.py` | Tool registry, manages all available tools |
| `factory.py` | Tool factory, supports **dynamic tool creation (LATM)** |
| `builtin/` | Built-in tool collection |

**Built-in Tools:**
| Tool | Description |
|------|-------------|
| `CodeExecutionTool` | Execute Python code in stateful kernel |
| `CodeAnalysisTool` | Analyze code structure |
| `LLMCallTool` | Sub-call LLM |
| `AnalyzeTextTool` | Text analysis |
| `ReadFileTool` | Read files |
| `WriteFileTool` | Write files |
| `ListDirectoryTool` | List directory contents |
| `ParseWordTool` | Parse Word documents |
| `ParseExcelTool` | Parse Excel spreadsheets (supports xlsx and xls) |
| `ParsePowerPointTool` | Parse PowerPoint presentations |
| `ParsePDFTool` | Parse PDF documents |

#### 3. executor/ - Code Execution Engine

| File | Description |
|------|-------------|
| `kernel.py` | **Stateful Python execution kernel** (variables persist across executions) |
| `state.py` | State manager, tracks variables, artifacts, and execution history |

**PythonKernel Features:**
- Variables persist across multiple executions
- Pre-loaded common libraries (math, numpy, pandas, etc.)
- Captures stdout/stderr
- Timeout support
- Module whitelist/blacklist control

#### 4. provider/ - LLM Provider Abstraction

| File | Description |
|------|-------------|
| `base.py` | Provider abstract base class and message types |
| `openai.py` | OpenAI/Azure OpenAI implementation |
| `anthropic.py` | Anthropic Claude implementation |

**Supported Providers:**
- OpenAI API
- Azure OpenAI
- Anthropic Claude
- Any OpenAI-compatible API

#### 5. memory/ - Memory System

| File | Description |
|------|-------------|
| `conversation.py` | Conversation memory (long-term), manages dialogue history |
| `working_memory.py` | Working memory (short-term), manages task context |

**ConversationMemory Features:**
- Stores conversation turns
- Supports multimodal messages (images)
- Auto-trims history exceeding max_turns
- Converts to LLM message format

**WorkingMemory Features:**
- Task stack management (main task/subtasks)
- Intermediate result tracking
- Item expiration support (TTL)
- Auto-cleanup of old items

#### 6. metacognition/ - Metacognition System

| File | Description |
|------|-------------|
| `capability.py` | Task capability assessment, analyzes task characteristics |
| `decision.py` | Computation mode decision (selects Neural/Symbolic/Hybrid/Tool mode) |

**Task Types:**
- COMPUTATION - Mathematical calculations
- DATA_ANALYSIS - Data processing and analysis
- REASONING - Logical reasoning
- CREATIVE - Creative tasks
- INFORMATION - Information retrieval
- CODE_GENERATION - Code generation
- CONVERSATION - General conversation

**Computation Modes:**
- NEURAL - Pure LLM reasoning
- SYMBOLIC - Code execution
- HYBRID - Combined approach
- TOOL - Use specific tools

#### 7. planner/ - Task Planning System

| File | Description |
|------|-------------|
| `decomposer.py` | Task decomposer, breaks complex tasks into subtasks |

**Task Data Structures:**
```python
@dataclass
class SubTask:
    id: str
    description: str
    status: TaskStatus  # PENDING, IN_PROGRESS, COMPLETED, FAILED, BLOCKED
    depends_on: List[str]
    suggested_approach: str
    suggested_tools: List[str]

@dataclass
class Task:
    description: str
    subtasks: List[SubTask]
    complexity: float
```

#### 8. config/ - Configuration Management

| File | Description |
|------|-------------|
| `settings.py` | All configuration class definitions |

**Configuration Hierarchy:**
```python
Settings                    # Main configuration container
├── ProviderConfig         # LLM provider config (type, api_key, model, etc.)
├── ExecutionConfig        # Code execution config (mode, timeout, allowed_modules)
└── MemoryConfig           # Memory config (max_turns, max_items, persist_tools)
```

---

## Core Workflow

### Think-Act Loop

```
User Input
    │
    ▼
┌─────────────────────────────────────┐
│  HybridAgent.run()                  │
│                                     │
│  while iterations < max_iterations: │
│      │                              │
│      ▼                              │
│  ┌─────────────┐                    │
│  │   THINK     │ ← LLM reasoning    │
│  │  (neural)   │   returns Action   │
│  └─────────────┘                    │
│      │                              │
│      ▼ Action                       │
│  ┌─────────────┐                    │
│  │    ACT      │ ← Execute action   │
│  │ (symbolic)  │   returns Result   │
│  └─────────────┘                    │
│      │                              │
│      ▼                              │
│  Check if complete/need response    │
│      │                              │
│      └───→ Update Context ─→ Loop   │
└─────────────────────────────────────┘
    │
    ▼
Final Response
```

### Tool Creation Flow (LATM)

```
User requests tool creation
    │
    ▼
LLM generates tool code
    │
    ▼
CodeValidator.validate()
    ├── Syntax check
    ├── Blocked module check
    └── Dangerous function check
    │
    ▼
ToolFactory.create_from_code()
    ├── Execute code to get function
    ├── Auto-detect parameters
    └── Create ToolSpec
    │
    ▼
GeneratedTool instance
    │
    ▼
ToolRegistry.register()
    │
    ▼
Persist to plugins/generated_tools.json
```

---

## Security Mechanisms

### Code Execution Sandbox

```python
# Default blocked modules
BLOCKED_MODULES = ["socket", "requests", "urllib", "http", ...]

# Blocked built-in functions
BLOCKED_BUILTINS = ["eval", "exec", "compile", "__import__", ...]

# Blocked attribute access
BLOCKED_ATTRIBUTES = ["__class__", "__bases__", "__globals__", ...]
```

### Tool Code Validation
- AST syntax analysis
- Module import checking
- Dangerous function call detection

### Execution Timeout
- Default 30 second timeout
- Configurable maximum memory limit

---

## Design Patterns

| Pattern | Application |
|---------|-------------|
| **Abstract Factory** | `BaseProvider` defines LLM call interface; `OpenAIProvider`/`AnthropicProvider` provide implementations |
| **Strategy** | `ComputationMode` (NEURAL/SYMBOLIC/HYBRID/TOOL); `ComputationDecider` selects strategy based on task |
| **Decorator** | `@tool` decorator converts regular functions to Tool objects |
| **Observer** (implicit) | `StateManager` tracks variable changes; `WorkingMemory` records execution process |
| **Command** | `Action` class encapsulates different action types; `ActionType` enum defines action types |

---

## Usage Examples

### Basic Chat

```python
import asyncio
from inception import HybridAgent, Settings

async def main():
    settings = Settings.from_env()
    agent = HybridAgent(settings=settings)

    response = await agent.chat("What is 15% of 250?")
    print(response)

asyncio.run(main())
```

### Data Analysis

```python
response = await agent.chat("""
Analyze this sales data:
Q1: $15,000
Q2: $22,000
Q3: $18,000
Q4: $25,000

Calculate total, average, and identify trends.
""")
```

### Tool Creation

The agent can create new tools when needed:

```python
response = await agent.chat("""
I need a tool that calculates compound interest.
Formula: A = P(1 + r/n)^(nt)
Where P=principal, r=rate, n=compounds per year, t=years
""")
```

---

## Configuration Options

### Provider Settings

| Option | Description | Default |
|--------|-------------|---------|
| `type` | Provider type (openai, anthropic, azure) | openai |
| `model` | Model name | gpt-4o |
| `api_key` | API key (or use env var) | - |
| `temperature` | Generation temperature | 0.7 |
| `max_tokens` | Max tokens per response | 4096 |

### Execution Settings

| Option | Description | Default |
|--------|-------------|---------|
| `mode` | sandbox or trusted | sandbox |
| `timeout` | Execution timeout (seconds) | 30 |
| `allowed_modules` | Whitelist of modules | [common safe modules] |
| `blocked_modules` | Blacklist of modules | [os, subprocess, etc.] |

---

## Extension Points

### Adding a New LLM Provider

1. Inherit from `BaseProvider`
2. Implement `complete()` and `complete_with_tools()` methods
3. Add initialization logic in `Settings.from_env()`

### Adding New Built-in Tools

1. Inherit from `Tool` class
2. Implement `spec` property and `execute()` method
3. Register in `register_builtin_tools()`

### Custom Memory Strategy

1. Modify `WorkingMemory._cleanup()` to adjust cleanup strategy
2. Implement `ConversationMemory.summarize()` to add conversation summarization

---

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

```bash
# Format code
black .

# Lint
ruff check .

# Type check
mypy inception/
```

---

## License

MIT License - see LICENSE file for details.

## Acknowledgments

This project draws inspiration from:
- [TaskWeaver](https://github.com/microsoft/TaskWeaver) - Stateful execution and planning
- [OpenCode](https://github.com/opencode-ai/opencode) - Agent architecture patterns
- The LATM (LLMs as Tool Makers) research paradigm
