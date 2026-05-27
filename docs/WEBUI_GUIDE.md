# MissionOrch-LC Web UI 指南

基于 **Streamlit** 的可视化前端，把 6-Agent 流水线 + 朴素 RAG 的所有运行时面板都铺出来。

---

## 🚀 一键启动

```bash
# 项目根目录
cd /path/to/MasterDegree/MissionOrch-LC
source .venv/bin/activate

python run_ui.py                  # 默认 http://localhost:8501
python run_ui.py --port 8888      # 换端口
python run_ui.py --host 0.0.0.0   # 对外开放
python run_ui.py --no-browser     # 不自动开浏览器（远程跑必备）
```

`run_ui.py` 会自动：
1. 把 `项目根 + src/` 加入 `PYTHONPATH`
2. 把项目根 `.env` 注入当前进程（让 `ModelRouter` 读到 API key）
3. 调 `streamlit.web.cli` 拉起 `webui/app.py`

---

## 📐 目录结构

```
webui/
├── app.py                         # 主入口（系统概览 + sidebar router）
├── state.py                       # session_state 强类型包装
├── components/
│   ├── runner.py                  # 后台线程跑 run_graph + 阶段事件队列
│   ├── rag_query.py               # RAG 直查同步封装
│   ├── rag_storage_inspector.py   # RAG 索引产物统计
│   ├── config_io.py               # YAML 读写（ruamel 保序 + 自动备份）
│   ├── env_io.py                  # .env 读写
│   └── ui_helpers.py              # 公共渲染辅助
└── pages/                         # Streamlit 多页面（自动加载）
    ├── 1_🚀_任务执行.py
    ├── 2_📊_结果详情.py
    ├── 3_🔍_RAG_直查.py
    ├── 4_⚙️_RAG_配置.py
    ├── 5_🔑_LLM_API_配置.py
    ├── 6_📚_知识库管理.py
    └── 7_📜_历史运行.py
```

---

## 🧭 7 个面板速览

| 面板 | 关键能力 |
|---|---|
| **🏠 主页（app.py）** | 系统快照：RAG 启用状态、已配置 LLM 数、API Keys 注入率、各 source 索引产物统计、最近一次运行 |
| **🚀 任务执行** | 输入 mission + 参数（max_iter / threshold / use_rag / legacy），后台线程跑 `run_graph`，6 个阶段实时步骤条 + 工具调用日志 |
| **📊 结果详情** | 6 Tab 分区：`mission_analysis` / `research_brief` / COA Markdown / **4 种结构化格式**（JSON/YAML/扁平/压缩）/ 验证 + Judge 历史 / Token + 计时；底部一键下载完整 JSON |
| **🔍 RAG 直查** | 选 source + mode → 调 `retrieve_with_rewrite` → 展示**原 query** + **改写后的 3 条子 query** + **按子 query 分组的命中片段**；保留最近 20 条历史 |
| **⚙️ RAG 配置** | 在线编辑 `config/rag.yaml`：总开关 / Embedding / Reranker / 解析器 / 知识源（增删改） / 检索 + Multi-Query；保存前预览原始 YAML，首次保存自动备份 `.bak` |
| **🔑 LLM API 配置** | 编辑 `config/models.yaml` 各 model_id（provider/base_url/model/temperature/timeout/max_retries）+ `.env` 中的 API keys + **每个模型独立 ping 测试**（发 1 token 验证连通） |
| **📚 知识库管理** | 各 source 索引产物概览表 → 选 source → 看现有源文件 + 上传新文件 + **跑索引**（force / 增量）+ 反查已入库文档 + 危险区清空索引 |
| **📜 历史运行** | 扫描 `output/*.json`，点选任一份用与"结果详情"一致的 7 Tab 视图复现展示 |

---

## 🔍 关键实现要点

### 1. 阶段流实时刷新

`components/runner.py` 在**后台线程**起 `asyncio` loop 跑 `run_graph`，主线程通过 `queue.Queue` 拉事件 + `st.rerun()` 轮询：

- `stage_start` 事件 → 步骤条某节点变绿
- `tool_start` / `tool_end` → 显示 Researcher 调 RAG 工具的 IO
- `done` / `error` → 终态写回 `session_state`

> Streamlit 的脚本是**整段重跑**模型，不能直接 `await`；用线程 + 队列是最稳的方案。

### 2. RAGManager 单例缓存

`@st.cache_resource` 缓存 `RAGManager`，避免每次进 RAG 直查页都重新加载本地 Qwen3 权重（CPU 模式下要 3-5s）。**保存 RAG 配置时自动 `cache_resource.clear()`**，下次进入会用新配置重建。

### 3. 朴素 RAG 验收口径（主页 + 知识库页都体现）

| 指标 | 朴素 RAG 期望值 |
|---|---|
| `vdb_chunks` | **> 0**（核心向量库） |
| `vdb_entities` / `vdb_relationships` | **= 0** |
| `graph_nodes` / `graph_edges` | **= 0** |
| `kv_store_full_docs` | = 已索引文件数 |

主页的 source 概览表会一目了然，方便随时核验"实体抽取确实被 monkey-patch 关掉了"。

### 4. 配置编辑器的安全性

- 用 `ruamel.yaml`（保序 + 注释保留）写盘，丢失的自定义字段最少
- **首次写盘自动备份** 到 `<file>.bak`，可手动 `mv .bak` 回滚
- API key 的 `${ENV_VAR:default}` 占位语法在编辑时**保留原文**，不会被展开后写死

---

## 🧪 推荐使用流程

```text
1. 首次：进 🔑 LLM API 配置 → 检查 .env API keys → 点 Ping 验证连通
2. 选项：进 ⚙️ RAG 配置 → 检查 embedding/reranker 模型路径 → 保存
3. 选项：进 📚 知识库管理 → 上传文件 → 点"索引"
4. 主流程：进 🚀 任务执行 → 输入 mission → 点"开始执行"
5. 看结果：进 📊 结果详情 → 7 个 Tab 翻看 → 下载完整 JSON
6. 调试 RAG：进 🔍 RAG 直查 → 单独看一条 query 的多路改写命中
```

---

## ❓ 常见问题

**Q1：启动报 `ModuleNotFoundError: missionorch_lc`**
A：请用 `python run_ui.py` 启动（它会自动注入 PYTHONPATH）。直接 `streamlit run webui/app.py` 也可以——`app.py` 顶部也做了 path 注入。

**Q2：RAG 直查页一直转圈**
A：首次会加载本地 Qwen3-Embedding（CPU 约 3-5s）+ Qwen3-Reranker（约 2s）。之后被 `@st.cache_resource` 缓住，再查就秒返回。

**Q3：API Ping 一直失败**
A：检查 `.env` 中对应的 `*_API_KEY` 是不是真的写进去了。Web UI 的"🔑 LLM API 配置"页保存后会**立即注入当前进程**，无需重启。

**Q4：保存了 rag.yaml 但 RAG 直查还是用旧配置**
A：保存按钮已经自动 `cache_resource.clear()`。如果还有问题，刷新一下浏览器即可。

**Q5：怎么禁用 Multi-Query？**
A：进 ⚙️ RAG 配置 → 「检索 + Multi-Query」Tab → 关掉 `query_rewrite.enabled` → 保存。

---

## 🔌 与后端的对应关系

| 前端模块 | 调用的后端 API |
|---|---|
| 🚀 任务执行 | `missionorch_lc.orchestrator_graph.run_graph` / `orchestrator.COAOrchestrator` |
| 🔍 RAG 直查 | `RAGManager.retrieve_with_rewrite` |
| ⚙️ RAG 配置 | 直接读写 `config/rag.yaml` |
| 🔑 LLM API 配置 | 直接读写 `config/models.yaml` + `.env`；ping 走 `ModelRouter.get(...).invoke([HumanMessage("ping")])` |
| 📚 知识库管理 | `scripts.index_knowledge_base._index_one_source` + `RAGManager.insert` |
| 📜 历史运行 | 直接扫 `output/*.json`，本地解析 |
