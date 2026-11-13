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

## 与 ScenClaw WebUI 联动

保留原 Streamlit／CLI 入口；独立 HTTP 服务复用 `run_graph` 的模型、RAG、profile 和评分配置，不跨仓库 import，也不改变默认规划算法。

在本仓库已有 Python 环境中启动（令牌由部署环境提供，不写入命令行或 URL）：

```bash
export MISSIONORCH_PLANNING_TOKEN="${PLANNING_SERVICE_TOKEN}"
PYTHONPATH=src python -m missionorch_lc.planning_server \
  --host 127.0.0.1 --port 8502 --data-dir data \
  --concurrency 1 --queue-limit 4 --job-timeout 1800
```

缺少令牌时拒绝启动。`data/planning-jobs/` 原子保存作业；重启后未完成作业标记 `interrupted`，不会自动重跑收费请求。并发数、等待队列和单作业超时可显式配置。跨机器部署应使用 HTTPS／受控反向代理；不要将明文 Bearer 服务直接暴露到公网。

接口：
- `GET /health`：无正文健康检查，不代表鉴权成功。
- `POST /v1/plans`：Bearer 鉴权，JSON `{ "request_id": "<UUID>", "mission": "<任务描述>" }`；同 ID 同内容返回原作业。
- `GET /v1/plans/<job_id>`、`POST /v1/plans/<job_id>/cancel`：查询／取消；取消正文为 `{}`。
- `POST /v1/coa/prepare`：JSON `{ "markdown": "<完整 Markdown>", "selected_task_ids": null }`；`null` 为整份，数组为选中行动。只解析与检查，不调用模型。

在 ScenClaw 根配置中启用 `planning`，并通过既有 `tools.ssrfWhitelist` 显式授权服务地址，详见 [ScenClaw README](../ScenClaw/README.md#任务规划与想定联动)。浏览器只连接 nanobot 的已鉴权 WebSocket；服务令牌不进入前端。

WebUI 支持自然语言生成、完整 Markdown 编辑、不可变版本、整份／部分选择及原聊天续修。附加实体／地理章节保留；部分选择仅提示选中文本／触发条件中的显式未选阶段或行动 ID，不自动补全前置行动。来源生成评分不等于编辑版评分，结构报告也不表示想定已通过仿真。

排查：`401` 检查两端令牌；`429 queue_full` 等待或显式调整容量；`timed_out` 检查原模型调用和服务超时；`interrupted` 检查已有作业／会话，显式发起新生成，不自动重放；`unparseable_plan`／`invalid_references` 修正矩阵、重复 ID、实体与地理引用。已保存规划在远端不可达时仍可由 ScenClaw 查看。

离线回归（fake runner 与临时 HTTP 服务，不调用模型或仿真）：

```bash
PYTHONPATH=src python -m pytest tests/test_planning.py -q
```

## 工具扩展

后续新增工具时，继承 `langchain_core.tools.BaseTool`（或 `@tool` 装饰器）即可，
并在 `tools/registry.py` 中注册，Agent 通过 `bind_tools` 直接启用 Function Calling。
