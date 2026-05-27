# MissionOrch-LC — 基于 LangChain + LangGraph 的多智能体 COA 编排系统

本项目是 [`MissionOrch`](../MissionOrch) 的 **LangChain + LangGraph 重构版**。
整体功能与原版一致（生成战役级 COA 矩阵），但底层全部改为 LangChain 生态，
并扩展为 **6-Agent 状态图流水线**（Analyst → Researcher → Planner ⇄ Judge ⇄ Reflector → Validator → Finalizer）。

## 测试通过情况

| 类别 | 结果 |
|---|---|
| **pytest 全量回归** | **72 / 72 通过**（16.84s） |
| **朴素 RAG 索引冒烟** | tmp_plain 3 文件 / 10 秒 / 0 实体 / 0 边 ✅ |
| **RAG 召回能力 showcase** | **7 / 7 命中**（doctrines 2/2 · tmp_plain 3/3 · glossary 2/2） |
| **Multi-Query 中文检索冒烟** | 3 / 3 命中（中文 query → 英文语料） |
| **6-Agent 端到端验证** | Judge 8.5 / 10 首轮 ACCEPT，COA 解析成功，4 格式输出齐全 |

跑测试：

```bash
pytest tests/ -q                                              # 单测
PYTHONPATH=src python scripts/smoke_naive_retrieve.py         # RAG 检索冒烟
PYTHONPATH=src python scripts/showcase_rag_recall.py          # RAG 召回 showcase
```

## 架构对照

| 层面 | 原版实现 | LangChain 重构版 |
| :--- | :--- | :--- |
| 模型路由 | 自研 `ModelRouter` + 5 个 Adapter | 复用 `ModelRouter`，统一返回 LangChain `ChatModel` |
| Agent 定义 | 自研 `BaseAgent.generate` | `ChatPromptTemplate \| ChatModel \| OutputParser`（LCEL） |
| JSON 解析 | 手写正则 `_extract_json` | `JsonOutputParser` / `with_structured_output` |
| 工具抽象 | 自研 `BaseTool` | `langchain_core.tools.BaseTool` |
| 编排循环 | Python `while` + 手动调用 | **LangGraph StateGraph**（条件路由 + 自动迭代） |
| RAG | `raganything` 封装 | **朴素 RAG**：parse→chunk→embedding，naive 检索，零实体抽取 |
| Query 改写 | 无 | **Multi-Query**：中→英翻译 + 术语扩展 + 同义改写，3 路并发检索 |
| 文档格式 | 仅 PDF | PDF（MinerU）/ DOCX / **HTML（bs4→md）** / MD / TXT |

## 目录结构

```
MissionOrch-LC/
├── src/missionorch_lc/
│   ├── core/
│   │   ├── rag/
│   │   │   ├── lightrag_factory.py    # LightRAG 朴素模式 monkey-patch
│   │   │   ├── html_to_md.py          # bs4 → Markdown 预处理器
│   │   │   ├── query_rewriter.py      # Multi-Query 改写
│   │   │   └── local_models.py        # 本地 Qwen3 Embedding/Reranker
│   │   ├── rag_manager.py             # 多 source 引擎管理 + 纯文本快速路径
│   │   ├── model_router.py            # LLM 路由
│   │   ├── coa_parser.py              # COA 矩阵解析
│   │   └── log_config.py
│   ├── agents/                        # Analyst / Researcher / Planner / Judge / Reflector / Validator
│   ├── tools/                         # LangChain Tool（RAG 检索 × N source）
│   ├── schemas/                       # pydantic 数据模型
│   ├── prompts/                       # 6 个 Agent 的提示词
│   └── orchestrator_graph.py          # LangGraph 6-Agent 状态图
├── config/                            # models.yaml / agents.yaml / rag.yaml
├── knowledge_base/                    # RAG 知识库（doctrines / maps / historical / glossary / tmp_plain）
├── scripts/                           # 索引 / 冒烟 / showcase
├── tests/                             # 72 测试用例
├── main.py                            # CLI 入口
└── docs/                              # ARCHITECTURE.md / RAG_GUIDE.md / README.md
```

## 安装

```bash
pip install -r requirements.txt
# 关键依赖：langchain / langgraph / lightrag-hku / raganything / mineru / beautifulsoup4 / torch / transformers
```

## 配置 API Key

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
export ARK_API_KEY=...           # 豆包
export DASHSCOPE_API_KEY=...     # 通义千问
export DEEPSEEK_API_KEY=...      # 默认主力（deepseek_v4_pro / deepseek_v4_flash / deepseek_chat_safe）
```

## 运行

```bash
# 6-Agent + 朴素 RAG + Multi-Query（默认）
python main.py --mission "Plan a SEAD strike against ..."

# 关掉 RAG（Researcher 节点降级）
python main.py --mission "..." --no-rag

# 索引知识库（PDF/DOCX 走 MinerU；MD/TXT/HTML 走纯文本快速路径，秒级入库）
python scripts/index_knowledge_base.py --source doctrines --force
```

更多用法请参考：

- `docs/README.md` —— 5 分钟跑通 + CLI 全参 + 编程接口
- `docs/RAG_GUIDE.md` —— 朴素 RAG 详解 + Multi-Query + 当前可召回内容清单
- `docs/ARCHITECTURE.md` —— 6-Agent 状态图 + 数据流

## 工具扩展

后续新增工具时，继承 `langchain_core.tools.BaseTool`（或 `@tool` 装饰器）即可，
并在 `tools/registry.py` 中注册，Agent 通过 `bind_tools` 直接启用 Function Calling。
