# RAG 使用指南 —— LightRAG + 本地 Qwen3 模型

> 本系统的 RAG 后端是 [HKUDS/RAG-Anything](https://github.com/HKUDS/RAG-Anything)
> （内部走 LightRAG）。Embedding / Reranker 走本地
> `/chatgpt_nas/dukaixuan.dkx/models/Qwen3-Embedding-0.6B` 与
> `Qwen3-Reranker-0.6B`，LLM（图谱抽取 / 摘要）走 ModelRouter。

## 1. 知识资源放在哪

```
MissionOrch-LC/
└── knowledge_base/
    ├── doctrines/      ← 作战条令、战术手册、训练大纲
    │     *.pdf *.docx *.md *.txt
    ├── maps/           ← 作战地图、地形态势（含图片）
    │     *.jpg *.jpeg *.png *.pdf
    ├── historical/     ← 历史战例、战后报告
    │     *.pdf *.txt *.md
    ├── glossary/       ← 术语 / 缩略语 / 装备代号（已带种子）
    │     seed_glossary.md
    └── rag_storage/    ← 自动生成的 LightRAG 持久化（KV/向量/图）
        ├── doctrines/
        ├── maps/
        ├── historical/
        └── glossary/
```

每个 source 在 `rag_storage/` 下有**独立 working_dir**，互不干扰；
新增 source 只需加目录 + 改 `config/rag.yaml`。

## 2. 一次性索引

```bash
# 索引所有 source
python scripts/index_knowledge_base.py

# 只索引一个
python scripts/index_knowledge_base.py --source doctrines

# 强制重建（忽略已索引文件签名）
python scripts/index_knowledge_base.py --force
```

脚本特性：
- 用文件名+size+mtime 做签名做增量
- 单个文件解析失败不会中断整批，只打 warning
- 索引产物写到 `rag_storage/<source>/`，下次启动自动复用

## 3. 关键配置：`config/rag.yaml`

```yaml
rag:
  enabled: true
  working_dir: "./knowledge_base/rag_storage"

  embedding:
    model_path: "/chatgpt_nas/dukaixuan.dkx/models/Qwen3-Embedding-0.6B"
    device: "auto"            # auto / cpu / cuda / cuda:0
    max_length: 8192
    batch_size: 8

  reranker:
    enabled: true
    model_path: "/chatgpt_nas/dukaixuan.dkx/models/Qwen3-Reranker-0.6B"

  llm_model_id: "seed_doubao"     # 用于实体抽取 / 知识图谱构建
  vision_model_id: null           # 想解析地图图片可设为 "qwen_vl"

  knowledge_sources:
    doctrines:
      path: "knowledge_base/doctrines"
      file_types: [".pdf", ".docx", ".md", ".txt"]
      description: "作战条例与战术手册"
    # ... 其他 sources

retrieval:
  search_mode: "hybrid"        # hybrid / local / global / naive
  top_k: 5
  rerank_top_n: 10
```

## 4. RAG 工具接入 Agent

`Researcher` 是 ReAct 风格的，能**主动**选用以下工具：

| 工具名 | 检索的 source | Researcher 何时调用 |
|---|---|---|
| `rag_doctrine_search` | doctrines | 需要权威规范 / 标准流程时 |
| `rag_map_search` | maps | 需要空间 / 地形参考时 |
| `rag_historical_search` | historical | 需要类比 / 经验教训时 |
| `rag_glossary_search` | glossary | 需要确认术语精确含义时 |

工具的实现见 `src/missionorch_lc/tools/rag_tools.py`。
工厂函数 `build_rag_tools(rag_manager)` 会**根据已配置的 source 自动**生成对应工具
（没在 `rag.yaml` 中配置的 source 不会暴露工具）。

## 5. 直接调用 RAG（绕过 Agent）

```python
import asyncio
from missionorch_lc.core.rag_manager import RAGManager

async def main():
    rag = RAGManager()                       # 读 config/rag.yaml
    print(rag.list_sources())                # ['doctrines', 'maps', ...]

    # 单源检索
    text = await rag.retrieve(
        "装甲集群在城市作战中的协同要点",
        source="doctrines",
        mode="hybrid",
    )
    print(text)

    # 多源并发检索
    all_results = await rag.retrieve_all(
        "夺取关键节点的常见战术",
        mode="hybrid",
    )
    for src, result in all_results.items():
        print(f"--- {src} ---\n{result}\n")

asyncio.run(main())
```

## 6. 本地模型工作机制

### Embedding（`Qwen3-Embedding-0.6B`）
- `padding_side="left"` + `last_token_pool`（取最后一个有效 token）
- 输出做 `F.normalize(p=2)` → 余弦距离 = 内积
- query 编码会自动加 `Instruct: {instruction}\nQuery: {text}` 前缀
- 进程级单例，多次 `build_local_embedding_func` 共享权重

### Reranker（`Qwen3-Reranker-0.6B`）
- 走 generative 风格：构造一个 yes/no 判别 prompt 喂给 CausalLM
- 取最后一个位置上 `yes`/`no` token 的 logit，做 2-class softmax
- `relevance_score = P(yes)`
- 默认 `top_n=10` 的截断

### LLM（图谱抽取 / 摘要）
- 通过 `ModelRouter.get_model(llm_model_id)` 拿到 LangChain ChatModel
- `_make_llm_callable_from_router` 把它包成 LightRAG 期望的
  `async def llm(prompt, system_prompt=None, history_messages=[], **kwargs) -> str`
- 因此 `llm_model_id` 必须支持普通对话（不需要 tool calling）

## 7. 与原项目（`MissionOrch/`）的差异

原项目的 RAG 用法是**错的**：
- `RAGManager.__init__` 只传了 `RAGAnythingConfig`，**没传** `llm_model_func` 和 `embedding_func` —— LightRAG 会因为缺必填参数无法工作
- `aquery()` 调用了不存在的参数 `top_k=...` 和 `vlm_enhanced=...`
- 完全没用上本地模型，默认依赖外网 OpenAI API

本仓库已经修复：
- 必填参数全部到位
- `aquery(query, mode=...)`，其他参数走 `lightrag_kwargs`
- Embedding / Reranker 全部走本地，LLM 走 ModelRouter

## 8. 常见问题

- **Q：第一次 `import` 很慢**
  A：transformers 在加载 0.6B 模型，CPU 上首次大约 10-20s；之后由进程级缓存。

- **Q：CUDA OOM**
  A：在 `rag.yaml` 把 `embedding.device: "cpu"` 或减小 `batch_size`。

- **Q：想换更大的嵌入模型**
  A：改 `embedding.model_path` 即可，结构兼容 Qwen 系列。注意 `embedding_dim`
  会自动从 `model.config.hidden_size` 读，下游不用改。

- **Q：想关掉 reranker**
  A：`rag.yaml` 里 `reranker.enabled: false`，会跳过 rerank，直接用 embedding 排序。

- **Q：LightRAG 索引太慢**
  A：是 LLM 抽取实体/关系慢；可在 `rag.yaml` 把 `llm_model_id` 换成更便宜更快的模型。
