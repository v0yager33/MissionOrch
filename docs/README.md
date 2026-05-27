# MissionOrch-LC 用户指南

> 基于 **LangChain + LangGraph** 的 6-Agent 战役级 COA（Course of Action）编排系统。
> 本目录还包含：
> - `ARCHITECTURE.md` —— 项目架构 / 数据流 / 状态机
> - `RAG_GUIDE.md` —— 朴素 RAG 路径、Multi-Query 改写、当前可召回内容清单

## 0. 当前测试通过情况（2026-05-06 最新）

| 维度 | 结果 |
|---|---|
| pytest 全量回归 | **72 / 72 通过**（16.84s） |
| 朴素 RAG 索引冒烟 | tmp_plain 3 文件 / 10 秒 / 0 实体 / 0 边 ✅ |
| RAG 召回 showcase | **7 / 7 命中**（doctrines 2/2 · tmp_plain 3/3 · glossary 2/2） |
| Multi-Query 中文 → 英文语料 | 3 / 3 命中 |
| 端到端 6-Agent | Judge 8.5/10 首轮 ACCEPT、coa_table 9574 字、4 格式输出齐全 |

## 1. 这个项目是什么

它把"指挥官给一段任务描述"到"产出可执行 COA 矩阵"的过程编排成一条
**有向状态图**，6 个 Agent 各司其职：

| 角色 | 输入 | 输出 | 默认模型 |
|---|---|---|---|
| **Analyst** 任务分析 | 自然语言任务 | JSON：意图/目标/实体/约束/研究问题 | `deepseek_v4_pro` |
| **Researcher** 知识研究 | 上一步 JSON + RAG 工具（ReAct 循环）| Markdown 研究简报 | `deepseek_chat_safe`（无 reasoning_content，避免 ReAct 多轮 400）|
| **Planner** 规划 | 任务 + 研究简报 + 反思 | COA 矩阵（Markdown 表格）| `deepseek_v4_pro` |
| **Judge** 评估 | COA 矩阵 | 0–10 分 × 5 维度 + 反馈 + verdict | `deepseek_v4_pro` |
| **Reflector** 反思 | COA + 反馈 | 改进建议 | `deepseek_v4_flash` |
| **Validator** 验证 | 解析后的 COA 对象 | is_valid + 矩阵指标 | `deepseek_v4_flash` |

> **Researcher 升级**：内置 **Multi-Query** 改写（原 query → 中文翻英文 + 术语扩展 + 同义改写
> 共 3 条并发检索后合并），中文 query 也能命中纯英文语料。详见 `RAG_GUIDE.md §3`。

整个流程 = `analyst → researcher → planner ⇄ judge ⇄ reflector → validator → finalize`，
其中 `planner ⇄ judge ⇄ reflector` 是迭代循环，直到分数达标或迭代上限。

## 2. 5 分钟跑通

```bash
# 0. 建议用 conda / venv
cd MissionOrch-LC
pip install -r requirements.txt

# 1. （可选）准备 RAG 知识库
#    把文献丢到 knowledge_base/<source>/，详见 docs/RAG_GUIDE.md
python scripts/index_knowledge_base.py

# 2. 跑！
python main.py --mission "在城市A区组织一次装甲集群清剿行动..."
```

不想接 RAG？加 `--no-rag` 即可，Researcher 自动降级。

## 3. CLI 全部参数

```
python main.py [--mission TEXT] [--no-rag] [--legacy]
               [--max-iter N] [--threshold X] [--output-json PATH]
```

| 参数 | 说明 | 默认 |
|---|---|---|
| `--mission TEXT` | 任务描述（不传则进入交互模式）| 无 |
| `--no-rag` | 跳过 Researcher 节点的 RAG 检索 | 启用 RAG |
| `--legacy` | 使用旧版 4-Agent 经典 Orchestrator（不推荐）| 走 6-Agent LangGraph |
| `--max-iter N` | 主循环最大迭代次数 | `agents.yaml` 中 `workflow.max_iterations` |
| `--threshold X` | early-stop 分数阈值 | `agents.yaml` 中 `workflow.quality_threshold` |
| `--output-json PATH` | 把完整结果写 JSON | 不写 |

## 4. 配置文件总览

| 文件 | 作用 |
|---|---|
| `config/models.yaml` | LLM 列表 + ModelRouter 路由策略 |
| `config/agents.yaml` | 每个 Agent 的 model_id / prompt_file / 阈值 |
| `config/rag.yaml` | RAG 总开关、本地模型路径、知识源 |

修改任何 yaml 不需要重启，下次 `python main.py` 自动生效。

## 5. 编程接口

最常用：

```python
import asyncio
from missionorch_lc.orchestrator_graph import run_graph

result = asyncio.run(run_graph(
    "在 X 区组织战术机动 ...",
    use_rag=True,
    max_iterations=3,
    quality_threshold=8.0,
))

print(result["coa_table"])           # COA Markdown
print(result["mission_analysis"])    # Analyst JSON
print(result["research_brief"])      # Researcher Markdown
print(result["validation"])          # Validator dict
print(result["token_usage"])         # 各模型 token 统计
```

也可拿到底层 graph，自己挂 callback：

```python
from missionorch_lc.orchestrator_graph import build_graph
graph, token_cb, timing_cb, rag_mgr = build_graph(use_rag=True)
state = await graph.ainvoke({"mission_input": "..."})
```

## 6. 观测性

- **LangSmith**：设置环境变量 `LANGSMITH_API_KEY` / `LANGSMITH_PROJECT`，
  自动 trace 每个 Agent 的 prompt / response / token；
- **本地日志**：`StageTimingCallback` 输出每阶段耗时，
  `TokenUsageCallback` 按 model 拆分 token；
- **stage_log**：`result["stage_log"]` 是按时间序列记录的阶段事件，
  适合写性能分析脚本。

## 7. 常见问题

- **Q：报 "本地 Qwen3 embedding/reranker 需要 torch + transformers"**
  A：`pip install torch transformers accelerate`，CPU 也能跑（慢点）。
- **Q：报 "lightrag 未安装"**
  A：`pip install lightrag-hku`，详见 `RAG_GUIDE.md`。
- **Q：Analyst 总是返回兜底结果**
  A：99% 是 LLM 没返回合法 JSON，把 `agents.yaml` 里 analyst 的
  `temperature_override` 调到 0.0；或换更强的 model_id。
- **Q：Researcher 不调用任何 RAG 工具就直接出简报**
  A：检查 `agents.yaml` 中 researcher 的 `use_rag: true`，且 `rag.yaml`
  中 `enabled: true`，并确保有真实数据被索引（看 `rag_storage/<source>/`）。
