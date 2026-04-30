# Knowledge Base —— RAG 知识库根目录

把领域文献按 **source 分类** 放到对应子目录里，再运行
`python scripts/index_knowledge_base.py` 进行一次性索引。索引产物落在
`rag_storage/<source>/` 下，下次启动复用。

## 目录结构

```
knowledge_base/
├── doctrines/    # 作战条令、战术手册、训练大纲
│   *.pdf *.docx *.md *.txt
├── maps/         # 作战地图、地形态势（含图片）
│   *.jpg *.png *.pdf
├── historical/   # 历史战例、战后报告
│   *.pdf *.txt *.md
├── glossary/     # 术语 / 缩略语 / 装备代号
│   *.md *.csv *.txt
└── rag_storage/  # 自动生成，存 LightRAG KV/Vector/Graph
    ├── doctrines/
    ├── maps/
    ├── historical/
    └── glossary/
```

## 添加新知识源（5 步）

1. 在本目录新建子文件夹 `knowledge_base/<your_source>/`
2. 把文件丢进去（支持类型见 `config/rag.yaml`）
3. 在 `config/rag.yaml` 的 `knowledge_sources` 中新增条目
4. 若想让 Researcher Agent 能"主动调用"新源，请在
   `src/missionorch_lc/tools/rag_tools.py` 的 `candidate_classes` 中添加
   一个继承 `_SourceRAGTool` 的工具类
5. 重新索引：`python scripts/index_knowledge_base.py --source <your_source>`

## 索引说明

- 每个 source 独立 `working_dir`，互不影响
- 增量：默认对已索引文件名做 hash 跳过；强制重建用 `--force`
- 图片在 `maps/` 下会经 `vision_model_func` 解析成文本（需要在 `rag.yaml`
  设置 `vision_model_id`）
