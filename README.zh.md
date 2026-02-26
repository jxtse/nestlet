# Nestlet

轻量级神经-符号 AI 智能体，支持递归推理与自主工具创建。

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![English](https://img.shields.io/badge/lang-English-blue.svg)](README.md)

## Nestlet 的独特之处

**递归神经-符号架构**：传统智能体是扁平结构——神经层（LLM）调用符号层（工具）就结束了。而 Nestlet 支持无限嵌套：符号工具可以调用神经推理，神经推理又可以调用更多符号工具，层层递进。

```
传统智能体:                Nestlet:

Neural ──► Symbolic       Neural ──► Symbolic ──► Neural ──► Symbolic ──► ...
   │                         │                        │
   └── 结束                   └── 可继续递归            └── 可继续递归
```

**自主工具创建 (LATM)**：当现有工具不足以完成任务时，Nestlet 会基于 LLMs-as-Tool-Makers 范式自动创建新工具。

## 快速开始

```bash
# 安装
git clone https://github.com/jxtse/nestlet.git
cd nestlet
pip install -e .

# 配置
cp config.example.yaml config.yaml
# 编辑 config.yaml 填入你的设置

# 设置 API Key
export OPENAI_API_KEY="your-api-key"

# 运行
python main.py
```

## 使用方式

### 交互模式

```bash
python main.py
```

### 单次任务

```bash
python main.py --task "计算 100 的阶乘"
```

### 作为库使用

```python
import asyncio
from nestlet import HybridAgent, Settings

async def main():
    agent = HybridAgent(settings=Settings.from_env())
    response = await agent.chat("分析这些数据并创建一个可视化工具")
    print(response)

asyncio.run(main())
```

## 核心概念

| 概念 | 说明 |
|------|------|
| **Think-Act 循环** | LLM 推理 → 决定动作 → 执行 → 更新上下文 → 循环 |
| **嵌套推理** | 工具可以调用 LLM，实现多层级问题分解 |
| **工具工厂** | 运行时动态创建、验证、注册新工具 |
| **有状态内核** | Python 执行环境，变量跨调用持久化 |

## 架构概览

```
inception/
├── agent/          # Think-Act 循环、上下文管理
├── tool/           # 工具注册表、工厂、内置工具
├── executor/       # 沙箱化 Python 内核
├── provider/       # LLM 抽象层 (OpenAI, Anthropic, Azure)
├── memory/         # 对话记忆 + 工作记忆
└── metacognition/  # 任务分析、计算模式选择
```

→ 详细模块文档请参阅 [ARCHITECTURE.md](ARCHITECTURE.md)

## 配置

复制示例配置并自定义：

```bash
cp config.example.yaml config.yaml
```

```yaml
provider:
  type: openai          # openai | anthropic | azure
  model: gpt-4o
  temperature: 0.7

execution:
  mode: sandbox         # sandbox | trusted
  timeout: 30.0
```

或使用环境变量：

```bash
export INCEPTION_PROVIDER=openai
export INCEPTION_MODEL=gpt-4o
export OPENAI_API_KEY=sk-...
```

## 支持的 LLM 提供商

- OpenAI (GPT-4, GPT-4o 等)
- Anthropic (Claude 3.5, Claude 4)
- Azure OpenAI
- 任何 OpenAI 兼容 API

## 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest tests/

# 格式化 & 检查
black . && ruff check .
```

## 致谢

本项目受以下项目启发：
- [Claude Code](https://claude.ai/code) / [OpenCode](https://github.com/opencode-ai/opencode) — CLI 智能体模式
- [TaskWeaver](https://github.com/microsoft/TaskWeaver) — 有状态执行
- [LATM](https://arxiv.org/abs/2305.17126) — LLMs as Tool Makers 范式

## 许可证

MIT
