# RAG 使用指南 —— 朴素 RAG + Multi-Query + 本地 Qwen3 模型

> 本系统的 RAG 后端是 [HKUDS/RAG-Anything](https://github.com/HKUDS/RAG-Anything)
> （内部走 LightRAG），但**已被改造为「朴素 RAG」模式**：
> - **索引**只做 `parse → chunk → embedding`，**完全不抽实体/关系/摘要**
> - **检索**只走 `mode="naive"` 向量召回（可选 Qwen3-Reranker），不做图遍历
> - **Multi-Query** 改写在工具层做：原 query + 中→英翻译 + 术语扩展 = 3 路并发检索后合并
>
> Embedding / Reranker 走本地
> `/chatgpt_nas/dukaixuan.dkx/models/Qwen3-Embedding-0.6B` 与
> `Qwen3-Reranker-0.6B`；LLM（仅用于 Multi-Query 改写、不再用于图谱抽取）走 ModelRouter。

## 0. 当前可召回内容清单（2026-05-06 实测 7/7 命中）

跑 `PYTHONPATH=src python scripts/showcase_rag_recall.py` 复现：

| # | source | 中文 query | 召回长度 | 命中关键词 |
|---|---|---|---:|---|
| 1 | `doctrines` | 作战原则中的『集中』和『目标』分别强调什么？ | 7914 字 | mass / objective / concentrat / principle |
| 2 | `doctrines` | 空中战役规划的关键步骤是什么？ | 2659 字 | air campaign / planning / step / phase |
| 3 | `tmp_plain` | EA-18G Growler 的核心任务 | 5801 字 | ea-18g / growler / jamm / electronic |
| 4 | `tmp_plain` | SEAD 的三个阶段 suppress / deceive / strike | 6939 字 | suppress / deceive / strike |
| 5 | `tmp_plain` | 沙漠风暴行动里的 SEAD 战法 | 4902 字 | desert storm / wild weasel / harm / ef-111 / ea-6b |
| 6 | `glossary` | SEAD 是什么意思？ | 1891 字 | sead / suppress / air defense |
| 7 | `glossary` | HARM 反辐射导弹的定义 | 625 字 | harm / anti-radiation / missile / agm-88 |

每条 query 都被 Multi-Query 拆成 3 条互补 sub-query，命中后 snippet 头部带
`> Retrieved via multi-query rewrite (3/3 sub-queries hit)` 标注。

**已索引语料现状**：

| source | 文件 | chunks 大小 | 模式 | 备注 |
|---|---|---:|---|---|
| `doctrines` | Volume-1-Basic-Doctrine.pdf + 1 HTML | 2.2 MB | 朴素*（带旧图副产物） | r7b 阶段索引时 monkey-patch 尚未生效，残留 5602 nodes/edges 不影响 naive 检索 |
| `glossary` | seed_glossary.md | 10 KB | ✅ 朴素 | 干净，0 实体 / 0 关系 |
| `tmp_plain` | sead_phases.md / sead_history.txt / sead_doctrine.html | 28 KB | ✅ 朴素 | 冒烟测试语料，0 实体 / 0 关系 |
| `maps` | （空目录） | – | – | 待填充 |
| `historical` | （空目录） | – | – | 待填充 |

## 1. 知识资源放在哪

```
MissionOrch-LC/
└── knowledge_base/
    ├── doctrines/      ← 作战条令、战术手册、训练大纲
    │     *.pdf *.docx *.doc *.html *.htm *.md *.txt
    ├── maps/           ← 作战地图、地形态势（含图片）
    │     *.jpg *.jpeg *.png *.pdf
    ├── historical/     ← 历史战例、战后报告
    │     *.pdf *.docx *.html *.txt *.md
    ├── glossary/       ← 术语 / 缩略语 / 装备代号（已带种子）
    │     seed_glossary.md
    ├── tmp_plain/      ← 朴素 RAG 冒烟测试语料（3 个小文件）
    │     sead_phases.md  sead_history.txt  sead_doctrine.html
    └── rag_storage/    ← 自动生成的 LightRAG 持久化（KV/向量）
        ├── doctrines/
        ├── maps/
        ├── historical/
        ├── glossary/
        └── tmp_plain/
```

每个 source 在 `rag_storage/` 下有**独立 working_dir**，互不干扰；
新增 source 只需加目录 + 改 `config/rag.yaml`。

### 各文件类型走的解析链路

| 扩展名 | 链路 | 速度（CPU 机器） |
|---|---|---|
| `.md` / `.txt` / `.csv` | 直接 `LightRAG.ainsert(text)` 纯文本快速路径 | **秒级** |
| `.html` / `.htm` / `.xhtml` | `bs4 → markdown` 后走纯文本路径（保留标题/列表/表格/链接/加粗） | **秒级** |
| `.pdf` | MinerU pipeline 解析 → chunk → `ainsert` | LayoutLM+OCR，每 PDF 几分钟 |
| `.docx` / `.doc` / `.ppt` / `.pptx` / `.xls` / `.xlsx` | MinerU 先 libreoffice 转 PDF，再走 PDF 路径 | 同上，需本地装 libreoffice |
| `.jpg` / `.png` / `.jpeg` | MinerU OCR | 分钟级 |

> **为什么要为纯文本做快速路径**：MinerU 即使对 .md/.txt 也会先 ReportLab 转 PDF
> 再启 MinerU FastAPI 子进程做 LayoutLM+OCR；CPU 机器上慢得不可用且毫无信息增益。
> 所以 `RAGManager._insert_plaintext` 直接调底层 `LightRAG.ainsert(text)` 绕开 MinerU。

## 2. 一次性索引

```bash
# 索引所有 source
python scripts/index_knowledge_base.py

# 只索引一个
python scripts/index_knowledge_base.py --source doctrines

# 强制重建（忽略已索引文件签名）
python scripts/index_knowledge_base.py --source doctrines --force
```

脚本特性：
- 用文件名+size+mtime 做签名做增量
- 单个文件解析失败不会中断整批，只打 warning
- 索引产物写到 `rag_storage/<source>/`，下次启动自动复用
- 内部统一调 `RAGManager.insert`，按扩展名自动选纯文本/MinerU 双路径

### 索引产物核验

朴素 RAG 模式下，每个 source 的 `rag_storage/<source>/` 应该长这样：

```
vdb_chunks.json              ← 必须非空（chunk × embedding，几 KB 起）
vdb_entities.json            ← ~49B 空骨架  ✅ 零实体
vdb_relationships.json       ← ~49B 空骨架  ✅ 零关系
graph_chunk_entity_relation.graphml  ← ~311B 空图 (0 nodes / 0 edges) ✅
kv_store_text_chunks.json    ← chunk 原文
kv_store_full_docs.json      ← 文档原文
```

索引日志里应该看到：
```
[naive_rag] skip extract_entities: N chunks   ← 实体抽取被 monkey-patch 拦下
Writing graph with 0 nodes, 0 edges            ← 图为空
已索引 ... → source=xxx (plaintext, NNN bytes) ← 纯文本快速路径
```

## 3. Multi-Query 改写（朴素 RAG 的"召回粘合剂"）

朴素向量检索没有图遍历兜底，单条 query 召回失败就真的没了。本系统在工具层加了
**Multi-Query 改写**（`src/missionorch_lc/core/rag/query_rewriter.py`）：

```
原 query：       "SEAD 的三个阶段是什么？"
                          │
            LLM 一次调用（deepseek_v4_flash）
                          ▼
sub-query #1:  three phases of SEAD                              ← 中→英翻译
sub-query #2:  Suppression of Enemy Air Defenses phase breakdown ← 术语扩展
sub-query #3:  electronic attack, destruction, and suppression…  ← 同义改写
                          │
              3 路并发 naive 检索 + 合并
                          ▼
最终给 Agent 一段带『Retrieved via multi-query rewrite (3/3 sub-queries hit)』标注的合并结果
```

调用入口（无论 Agent 用工具，还是直接编程）都走 `retrieve_with_rewrite`：

```python
text = await rag_manager.retrieve_with_rewrite(
    "中文 query 也能命中英文语料",
    source="doctrines",
    mode="naive",
)
```

带缓存（默认 256 条 LRU），同一条 query 不会重复改写。

## 4. 关键配置：`config/rag.yaml`

```yaml
rag:
  enabled: true
  working_dir: "./knowledge_base/rag_storage"

  # 解析配置（朴素 RAG 模式下其实只对 PDF/DOCX/图片有效）
  parser: "mineru"
  enable_image_processing: true
  enable_table_processing: true
  enable_equation_processing: false

  embedding:
    model_path: "/chatgpt_nas/dukaixuan.dkx/models/Qwen3-Embedding-0.6B"
    device: "auto"            # auto / cpu / cuda / cuda:0
    max_length: 8192
    batch_size: 8

  reranker:
    enabled: true
    model_path: "/chatgpt_nas/dukaixuan.dkx/models/Qwen3-Reranker-0.6B"

  # ⚠️ 注意：朴素 RAG 模式下，LLM 仅用于 Multi-Query 改写
  # 实体抽取已被 monkey-patch 关闭，所以这里设个轻量模型即可
  llm_model_id: "deepseek_v4_flash"
  vision_model_id: null

  knowledge_sources:
    doctrines:
      path: "knowledge_base/doctrines"
      file_types: [".pdf", ".docx", ".doc", ".html", ".htm", ".md", ".txt"]
      description: "作战条例与战术手册"
    # ... 其他 sources

retrieval:
  search_mode: "naive"         # 朴素 RAG 固定走 naive
  top_k: 5
  score_threshold: 0.6
  rerank_top_n: 10

  # Multi-Query 改写
  query_rewrite:
    enabled: true
    llm_model_id: "deepseek_v4_flash"
    num_queries: 3             # 含原 query 的总条数
    cache_size: 256
    temperature: 0.2
    max_tokens: 400
    max_per_section_chars: 2500
```

## 5. RAG 工具接入 Agent

`Researcher` 是 ReAct 风格的，能**主动**选用以下工具：

| 工具名 | 检索的 source | Researcher 何时调用 |
|---|---|---|
| `rag_doctrine_search` | doctrines | 需要权威规范 / 标准流程时 |
| `rag_map_search` | maps | 需要空间 / 地形参考时 |
| `rag_historical_search` | historical | 需要类比 / 经验教训时 |
| `rag_glossary_search` | glossary | 需要确认术语精确含义时 |

工具的实现见 `src/missionorch_lc/tools/rag_tools.py`，**底层全部走
`retrieve_with_rewrite`（带 Multi-Query）**，所以 Researcher 不需要自己换不同
说法反复检索，工具会自动给 3 路改写。

工厂函数 `build_rag_tools(rag_manager)` 会**根据已配置的 source 自动**生成对应工具
（没在 `rag.yaml` 中配置的 source 不会暴露工具）。

## 6. 直接调用 RAG（绕过 Agent）

```python
import asyncio
from missionorch_lc.core.rag_manager import RAGManager

async def main():
    rag = RAGManager()                       # 读 config/rag.yaml
    print(rag.list_sources())                # ['doctrines', 'maps', ...]

    # 单源检索（带 Multi-Query 改写，推荐）
    text = await rag.retrieve_with_rewrite(
        "装甲集群在城市作战中的协同要点",
        source="doctrines",
        mode="naive",
    )
    print(text)

    # 不要 Multi-Query，直接单条向量检索
    text2 = await rag.retrieve(
        "MOPP level 4 protective posture",
        source="doctrines",
        mode="naive",
    )

    # 多源并发检索
    all_results = await rag.retrieve_all(
        "SEAD execution case study",
        mode="naive",
    )
    for src, result in all_results.items():
        print(f"--- {src} ---\n{result}\n")

asyncio.run(main())
```

## 7. 本地模型工作机制

### Embedding（`Qwen3-Embedding-0.6B`）
- `padding_side="left"` + `last_token_pool`（取最后一个有效 token）
- 输出做 `F.normalize(p=2)` → 余弦距离 = 内积
- query 编码会自动加 `Instruct: {instruction}\nQuery: {text}` 前缀
- 进程级单例，多次 `build_local_embedding_func` 共享权重
- 输出维度 1024（从 `model.config.hidden_size` 自动读，下游 vdb 自动适配）

### Reranker（`Qwen3-Reranker-0.6B`）
- 走 generative 风格：构造一个 yes/no 判别 prompt 喂给 CausalLM
- 取最后一个位置上 `yes`/`no` token 的 logit，做 2-class softmax
- `relevance_score = P(yes)`
- 默认 `top_n=10` 截断
- 在朴素 RAG 中作为 naive 召回的二次排序

### LLM（仅用于 Multi-Query 改写）
- 通过 `ModelRouter.get_model(llm_model_id)` 拿到 LangChain ChatModel
- `_make_llm_callable_from_router` 把它包成 LightRAG 期望的
  `async def llm(prompt, system_prompt=None, history_messages=[], **kwargs) -> str`
- 朴素 RAG 模式下 LLM 不参与索引（实体抽取已被 monkey-patch 关闭），
  所以 `llm_model_id` 用 `deepseek_v4_flash` 这种轻量快模型就够

## 8. 朴素 RAG 是怎么实现的（实现细节）

### Monkey-patch 4 点（`core/rag/lightrag_factory.py`）

模块 import 时立即 patch，**关闭 LightRAG 索引时所有 LLM 驱动的图构建步骤**：

| Patch 点 | 替换为 | 效果 |
|---|---|---|
| `LightRAG._process_extract_entities` | no-op，返回 `[]` | 跳过实体抽取总入口 |
| `lightrag.operate.extract_entities` | no-op | 兜底底层实现 |
| `lightrag.operate.merge_nodes_and_edges` | no-op | 跳过节点/边合并 |
| `lightrag.lightrag` 模块的 import 引用 | 同步替换 | 防止某些代码路径绕开前面的 patch |

加上 `entity_extract_max_gleaning=0` 关掉 gleaning 轮数。

### 纯文本快速路径（`core/rag_manager.py::_insert_plaintext`）

`.md/.txt/.csv` 直接读文件 → `LightRAG.ainsert(text, file_paths=...)`，绕开 MinerU。
HTML 则用 `core/rag/html_to_md.py` 先 bs4 转 md，再走纯文本路径。

### Multi-Query 改写器（`core/rag/query_rewriter.py`）

- 单次 LLM 调用，要求模型输出 `<query>...</query>` 包裹的 N-1 条改写
- `_parse_queries` 容错解析（缺标签 / 空行 / 编号都能容忍）
- LRU 缓存避免相同 query 重复改写
- 失败兜底：返回 `[原 query]`，确保检索一定能进行

## 9. 与原项目（`MissionOrch/`）的差异

| 维度 | 原项目（`MissionOrch/`） | 本项目（`MissionOrch-LC/`） |
|---|---|---|
| RAGManager 参数 | 只传 `RAGAnythingConfig`，缺 `llm_model_func/embedding_func` 直接报错 | ✅ 必填参数全到位 |
| `aquery` 调用 | 用了不存在的参数 `top_k=` / `vlm_enhanced=` | ✅ `aquery(query, mode=...)`，其它走 `lightrag_kwargs` |
| 本地模型 | 完全没用上，默认依赖外网 OpenAI API | ✅ Embedding/Reranker 全本地 |
| 索引模式 | 默认全图谱模式（LLM 抽实体/关系，慢且贵） | ✅ 朴素模式（chunk+embedding，秒级） |
| 跨语言检索 | 单条 query，命中靠运气 | ✅ Multi-Query 3 路改写，中文 query 也能命中英文语料 |
| HTML 支持 | MinerU pipeline 不支持 HTML | ✅ bs4→md 预处理 |

## 10. 常见问题

- **Q：第一次 `import` 很慢**
  A：transformers 在加载 0.6B 模型，CPU 上首次大约 10-20s；之后由进程级缓存。

- **Q：CUDA OOM**
  A：在 `rag.yaml` 把 `embedding.device: "cpu"` 或减小 `batch_size`。

- **Q：想换更大的嵌入模型**
  A：改 `embedding.model_path` 即可，结构兼容 Qwen 系列。`embedding_dim`
  会自动从 `model.config.hidden_size` 读，下游不用改。

- **Q：想关掉 reranker**
  A：`rag.yaml` 里 `reranker.enabled: false`，会跳过 rerank，直接用 embedding 排序。

- **Q：想关掉 Multi-Query 改写**
  A：`rag.yaml` 里 `retrieval.query_rewrite.enabled: false`；
  或者直接调 `RAGManager.retrieve(...)`（不带 `_with_rewrite` 后缀）。

- **Q：LightRAG 索引太慢**
  A：本系统已切朴素模式（不抽实体/关系），单文件入库应在秒级（纯文本）或
  几分钟（PDF MinerU）。如果还慢，看是不是 PDF 卡在 MinerU LayoutLM。

- **Q：怎么验证朴素模式生效了？**
  A：索引日志里应该有 `[naive_rag] skip extract_entities: N chunks` 和
  `Writing graph with 0 nodes, 0 edges`；`rag_storage/<source>/vdb_entities.json`
  约 49 字节（空骨架）。

- **Q：MinerU 太慢，能不能完全不用？**
  A：能。把 PDF 提前转成 md（用任何工具）放到 `knowledge_base/<source>/`，
  会自动走纯文本快速路径，秒级入库。
