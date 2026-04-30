# MissionOrch-LC — 基于 LangChain 的多智能体 COA 编排系统

本项目是 [`MissionOrch`](../MissionOrch) 的 **LangChain 重构版**。整体功能与原版一致（生成战役级 COA 矩阵），但底层全部改为 LangChain 生态。

## 架构对照

| 层面 | 原版实现 | LangChain 重构版 |
| :--- | :--- | :--- |
| 模型路由 | 自研 `ModelRouter` + 5 个 Adapter | `ChatOpenAI / ChatAnthropic / ChatGoogleGenerativeAI` |
| Agent 定义 | 自研 `BaseAgent.generate` | `ChatPromptTemplate \| ChatModel \| OutputParser`（LCEL） |
| JSON 解析 | 手写正则 `_extract_json` | `JsonOutputParser` |
| 工具抽象 | 自研 `BaseTool` | `langchain_core.tools.BaseTool` |
| 编排循环 | Python `while` + 手动调用 | Python `while` + LCEL Runnable（async invoke） |
| RAG | `raganything` 封装 | 保留 `RAGManager`，以 LangChain Tool 形式暴露 |

## 目录结构

```
MissionOrch-LC/
├── src/missionorch_lc/
│   ├── core/              # 模型路由、RAG、日志、COA 解析/转换
│   ├── agents/            # Planner / Judge / Reflector / Validator（LCEL chain）
│   ├── tools/             # LangChain Tool（RAG 检索、COA 语法校验）
│   ├── schemas/           # pydantic 数据模型（COA / JudgeResult / ValidationResult）
│   ├── prompts/           # 4 个 Agent 的提示词模板（.txt）
│   └── orchestrator.py    # 规划-评估-反思循环编排
├── config/                # models.yaml / agents.yaml / rag.yaml
├── main.py                # CLI 入口
└── example_usage.py       # 使用示例
```

## 安装

```bash
pip install -r requirements.txt
```

## 配置 API Key

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
export ARK_API_KEY=...           # 豆包
export DASHSCOPE_API_KEY=...     # 通义千问
export DEEPSEEK_API_KEY=...
```

## 运行

```bash
python main.py
# 或
python example_usage.py
```

## 工具扩展

后续新增工具时，继承 `langchain_core.tools.BaseTool`（或 `@tool` 装饰器）即可，并在 `tools/registry.py` 中注册，Agent 可通过 `bind_tools` 直接启用 Function Calling。
