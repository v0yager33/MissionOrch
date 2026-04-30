# MissionOrch-LC 架构文档

## 1. 整体定位

把"自由形式军事任务描述"压缩为"结构化、可解析、可验证的 COA 矩阵"，
并保证整条链路可观测、可重放、可替换 LLM。

## 2. 模块分层

```
┌─────────────────────────────────────────────────────────────┐
│                      main.py (CLI)                          │
└────────────────────────┬────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────┐
│             orchestrator_graph.py (LangGraph)               │
│  StateGraph(COAGraphState):                                 │
│    analyst → researcher → planner ⇄ judge ⇄ reflector       │
│                                              → finalize     │
└────┬────────────────────────────────────────────────────────┘
     │ 调用
     ▼
┌─────────────────────────────────────────────────────────────┐
│                   agents/ (LangChain LCEL)                  │
│  BaseAgent → Analyst | Researcher | Planner | Judge |       │
│              Reflector | Validator                          │
└────┬────────────────────────────────────────────────────────┘
     │ bind_tools / aquery
     ▼
┌──────────────────┐  ┌────────────────────────────────────┐
│ tools/rag_tools  │  │ core/rag_manager + core/rag/*      │
│  - Doctrine      │  │  RAGManager → 多 source            │
│  - Map           │  │  build_rag_anything → LightRAG     │
│  - Historical    │  │  build_local_embedding_func        │
│  - Glossary      │  │  build_local_rerank_func           │
└──────────────────┘  └──────┬─────────────────────────────┘
                              │
                              ▼
                ┌──────────────────────────────────┐
                │  knowledge_base/<source>/        │ ← 原始文档
                │  knowledge_base/rag_storage/     │ ← 索引产物
                └──────────────────────────────────┘
```

## 3. 状态对象（COAGraphState）

```python
mission_input: str
mission_analysis: dict      # Analyst 产物
research_brief: str         # Researcher 产物
current_coa_text: str       # Planner 当前轮 COA Markdown
iteration: int              # 已完成的 plan→judge 轮数
history: list[dict]         # 每轮 score / verdict / feedback
reflection: str             # Reflector 给 Planner 的反思
final_coa_obj: COA | None   # Finalize 解析出的 Pydantic 对象
validation: dict            # Validator 结果
outputs: dict               # JSON / YAML / flat / condensed 多格式
token_usage / timing / stage_log
```

LangGraph 的状态是 **不可变 dict 风格**：每个节点只 return 想更新的字段，
框架做合并；不存在的字段保留原值。

## 4. 节点职责

| 节点 | 节点函数 | 关键行为 |
|---|---|---|
| `analyst` | `_make_analyst_node` | LCEL `prompt | model | JsonOutputParser`；失败则降级写最小 JSON |
| `researcher` | `_make_researcher_node` | 检测 RAG 是否启用；启用则 `bind_tools` 后 ReAct 工具循环 |
| `planner` | `_make_planner_node` | 把 `mission_analysis + research_brief` 拼成 `knowledge` 给 Planner |
| `judge` | `_make_judge_node` | 调用 `JudgeAgent.evaluate`；写入 history 和 reflection（暂存）|
| `reflector` | `_make_reflector_node` | 把 feedback 转成可执行改进建议 |
| `finalize` | `_make_finalize_node` | `COATableParser` 解析 → Validator 验证 → COATransformer 输出多格式 |

条件边 `_should_continue`：
- `score >= quality_threshold` 且 `early_stop=True` → finalize
- `iteration >= max_iterations` → finalize
- 否则 → reflector → planner

## 5. 模型路由（ModelRouter）

`core/model_router.py` 是一个轻量层，把"逻辑 model_id"映射到"具体 LangChain ChatModel 实例"：

- 按 `models.yaml` 读取 provider / api_base / temperature
- 自动套 `with_retry` + `with_fallbacks`（如 `seed_doubao` fallback 到 `qwen_max`）
- 对支持 structured output 的模型，BaseAgent 提供 `bind_model(structured_schema=...)`

## 6. RAG 子系统

详见 `RAG_GUIDE.md`。简述：

- `build_local_embedding_func` 把本地 `Qwen3-Embedding-0.6B` 包成 `EmbeddingFunc`
  （last-token pooling + L2 normalize，async 接口）
- `build_local_rerank_func` 把本地 `Qwen3-Reranker-0.6B` 包成异步 rerank 函数
  （generative yes/no 概率打分）
- `build_rag_anything` 在两个本地模型 + 一个 ModelRouter LLM 上构建 `RAGAnything`
- `RAGManager` 按 source 维度独立持有 `RAGAnything` 实例，每个 source 独立 working_dir
- `build_rag_tools` 把每个 source 包成一个 LangChain `BaseTool` 暴露给 Researcher

## 7. 容错策略

| 失败位置 | 兜底行为 |
|---|---|
| Analyst LLM 返回非 JSON | 用 mission 文本兜底，写最小 objective + research_query |
| Researcher 模型不支持 tool calling | 直接降级出简报模板 |
| Researcher 工具调用上限到了仍未收敛 | 强制收敛：去掉 tools 再 invoke 一次 |
| Planner 输出过短 | 抛 ValueError，由上层 try/except 接收 |
| Judge 失败 | 给 5.0 分 + 错误信息，流程不中断 |
| Reflector 失败 | 把 judge 的 feedback 原样塞进 reflection |
| Parser 失败 | 写 COA(description="Parse error") 的占位对象 |
| Validator 失败 | is_valid=False + 错误信息 |

## 8. 可观测性

- LangSmith：`AppSettings.tracing.activate()` 自动开
- `TokenUsageCallback`：按 `model_name` 累计 prompt/completion/total
- `StageTimingCallback`：监听 `on_chain_start/end`，按 `run_name` 计时
- `stage_log`：业务级阶段事件，方便做依赖分析、绘制 sankey

## 9. 扩展指南

### 新增一个知识源
1. `knowledge_base/<your_source>/` 放文件
2. `config/rag.yaml` 的 `knowledge_sources` 加条目
3. `tools/rag_tools.py` 加一个 `_SourceRAGTool` 子类 + 在 `candidate_classes` 注册

### 新增一个 Agent
1. `prompts/<your_agent>.txt` 写 prompt（用 `{var}` 占位符）
2. `agents/<your_agent>.py` 继承 `BaseAgent`，实现 `_build_prompt` + 业务方法
3. `agents/__init__.py` 导出
4. `config/agents.yaml` 加配置块
5. `orchestrator_graph.py` 加节点 + 改边

### 替换 LLM
不动代码，只改 `config/models.yaml`：
```yaml
seed_doubao:
  provider: openai_compatible
  api_base: ...
  api_key_env: ...
```
