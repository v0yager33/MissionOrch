"""RAG 配置编辑页 —— 在线编辑 config/rag.yaml。"""

from __future__ import annotations

from typing import Any, Dict

import streamlit as st

from webui.components.config_io import load_yaml, save_yaml
from webui.components.ui_helpers import render_header
from webui.state import init_state

init_state()
render_header(
    "⚙️ RAG 配置",
    "编辑 `config/rag.yaml`：启用开关 / 嵌入模型 / 重排序 / 解析器 / 知识源 / 检索 / Chunking / Multi-Query。",
)

CONFIG_PATH = "config/rag.yaml"

# 每次进入页面都从盘读取（确保不丢手动改动）
raw_cfg = load_yaml(CONFIG_PATH)
rag_cfg: Dict[str, Any] = raw_cfg.get("rag") or {}
retrieval_cfg: Dict[str, Any] = raw_cfg.get("retrieval") or {}
chunking_cfg: Dict[str, Any] = raw_cfg.get("chunking") or {}

st.info(f"当前文件：`{CONFIG_PATH}`（首次保存时会自动备份到 `.bak`）")

# ── Tab 分组 ──
tabs = st.tabs(
    [
        "🟢 总开关 / 模型",
        "📄 解析器",
        "📚 知识源",
        "🔎 检索",
        "✂️ Chunking",
        "🔁 Multi-Query",
        "📝 原始 YAML",
    ]
)


# ── Tab 1: 总开关 + Embedding + Reranker + LLM ──
with tabs[0]:
    rag_cfg["enabled"] = st.toggle(
        "启用 RAG（rag.enabled）",
        value=bool(rag_cfg.get("enabled", True)),
        help="关闭后 Researcher 节点降级，且 RAG 直查页不可用。",
    )
    rag_cfg["working_dir"] = st.text_input(
        "索引产物根目录（rag.working_dir）",
        value=str(rag_cfg.get("working_dir", "./knowledge_base/rag_storage")),
    )
    rag_cfg["llm_model_id"] = st.text_input(
        "索引/检索阶段使用的 LLM model_id（rag.llm_model_id）",
        value=str(rag_cfg.get("llm_model_id", "deepseek_v4_flash")),
        help="必须存在于 config/models.yaml 中。建议用非思考模式（响应里没有 reasoning_content）。",
    )
    rag_cfg["vision_model_id"] = st.text_input(
        "视觉模型 model_id（rag.vision_model_id，可选）",
        value=str(rag_cfg.get("vision_model_id") or ""),
    ) or None

    st.markdown("#### Embedding 模型（必填）")
    emb_cfg = rag_cfg.get("embedding") or {}
    emb_cfg["model_path"] = st.text_input(
        "本地 Qwen3-Embedding 模型路径",
        value=str(emb_cfg.get("model_path", "/chatgpt_nas/dukaixuan.dkx/models/Qwen3-Embedding-0.6B")),
    )
    e_col1, e_col2, e_col3 = st.columns(3)
    emb_cfg["device"] = e_col1.selectbox(
        "device", ["auto", "cpu", "cuda", "cuda:0"],
        index=["auto", "cpu", "cuda", "cuda:0"].index(str(emb_cfg.get("device", "auto"))),
    )
    emb_cfg["max_length"] = e_col2.number_input(
        "max_length", 256, 8192, int(emb_cfg.get("max_length", 2048)), step=256
    )
    emb_cfg["batch_size"] = e_col3.number_input(
        "batch_size", 1, 64, int(emb_cfg.get("batch_size", 4))
    )
    emb_cfg["instruction"] = st.text_area(
        "instruction（喂给 Qwen3 Embedding 的指令）",
        value=str(emb_cfg.get("instruction") or ""),
        height=80,
    )
    rag_cfg["embedding"] = emb_cfg

    st.markdown("#### Reranker 模型（可选）")
    rer_cfg = rag_cfg.get("reranker") or {}
    rer_cfg["enabled"] = st.toggle(
        "启用 Reranker（reranker.enabled）",
        value=bool(rer_cfg.get("enabled", True)),
    )
    rer_cfg["model_path"] = st.text_input(
        "本地 Qwen3-Reranker 模型路径",
        value=str(rer_cfg.get("model_path", "/chatgpt_nas/dukaixuan.dkx/models/Qwen3-Reranker-0.6B")),
    )
    r_col1, r_col2 = st.columns(2)
    rer_cfg["device"] = r_col1.selectbox(
        "device ",
        ["auto", "cpu", "cuda", "cuda:0"],
        index=["auto", "cpu", "cuda", "cuda:0"].index(str(rer_cfg.get("device", "auto"))),
        key="reranker_device",
    )
    rer_cfg["max_length"] = r_col2.number_input(
        "max_length ", 512, 16384, int(rer_cfg.get("max_length", 8192)), step=512,
        key="reranker_max_length",
    )
    rag_cfg["reranker"] = rer_cfg


# ── Tab 2: 解析器 ──
with tabs[1]:
    rag_cfg["parser"] = st.selectbox(
        "文档解析器（parser）",
        ["mineru", "docling"],
        index=["mineru", "docling"].index(str(rag_cfg.get("parser", "mineru"))),
    )
    rag_cfg["enable_image_processing"] = st.toggle(
        "图片处理（enable_image_processing）",
        value=bool(rag_cfg.get("enable_image_processing", True)),
    )
    rag_cfg["enable_table_processing"] = st.toggle(
        "表格处理（enable_table_processing）",
        value=bool(rag_cfg.get("enable_table_processing", True)),
    )
    rag_cfg["enable_equation_processing"] = st.toggle(
        "公式处理（enable_equation_processing）",
        value=bool(rag_cfg.get("enable_equation_processing", False)),
    )
    rag_cfg["parser_lang"] = st.text_input(
        "MinerU 解析语言（parser_lang）",
        value=str(rag_cfg.get("parser_lang", "en")),
        help="en / zh / ch_doc 等，影响 OCR 模型选择",
    )


# ── Tab 3: 知识源 ──
with tabs[2]:
    sources_cfg: Dict[str, Any] = dict(rag_cfg.get("knowledge_sources") or {})
    st.caption("每个 source 对应一个独立的 working_dir。删除 source 不会删除已索引产物。")

    # 列出已有 source 进行编辑
    new_sources: Dict[str, Any] = {}
    for source_name, source_cfg in sources_cfg.items():
        with st.expander(f"📁 {source_name}", expanded=False):
            keep = st.checkbox(f"保留 {source_name}", value=True, key=f"keep_{source_name}")
            if not keep:
                continue
            edit_path = st.text_input(
                "目录",
                value=str(source_cfg.get("path", "")),
                key=f"path_{source_name}",
            )
            edit_types = st.text_input(
                "支持的文件后缀（逗号分隔）",
                value=", ".join(source_cfg.get("file_types") or []),
                key=f"types_{source_name}",
            )
            edit_desc = st.text_input(
                "描述",
                value=str(source_cfg.get("description", "")),
                key=f"desc_{source_name}",
            )
            new_sources[source_name] = {
                "path": edit_path,
                "file_types": [
                    item.strip() for item in edit_types.split(",") if item.strip()
                ],
                "description": edit_desc,
            }

    # 新增 source
    with st.expander("➕ 新增 source", expanded=False):
        new_name = st.text_input("名称（英文小写）", value="", key="new_source_name")
        new_path = st.text_input("目录路径", value="knowledge_base/", key="new_source_path")
        new_types = st.text_input(
            "支持的文件后缀（逗号分隔）",
            value=".pdf, .docx, .md, .txt",
            key="new_source_types",
        )
        new_desc = st.text_input("描述", value="", key="new_source_desc")
        if st.button("加入新 source", key="add_new_source"):
            if not new_name.strip():
                st.error("名称不能为空")
            elif new_name in new_sources:
                st.error(f"已存在同名 source: {new_name}")
            else:
                new_sources[new_name] = {
                    "path": new_path,
                    "file_types": [
                        item.strip() for item in new_types.split(",") if item.strip()
                    ],
                    "description": new_desc,
                }
                st.success(f"已暂存 source `{new_name}`，记得点击下方『保存』。")
    rag_cfg["knowledge_sources"] = new_sources


# ── Tab 4: 检索 ──
with tabs[3]:
    st.caption(
        "**这些参数即时生效**（保存后下次 `RAGManager` 构建就会读到，不需要重建索引）。"
        " 改完记得点底部『保存』。"
    )

    retrieval_cfg["search_mode"] = st.selectbox(
        "检索模式（retrieval.search_mode）",
        ["naive", "hybrid", "local", "global"],
        index=["naive", "hybrid", "local", "global"].index(
            str(retrieval_cfg.get("search_mode", "naive"))
        ),
        help=(
            "朴素 RAG 推荐 naive（只走 chunk 向量召回）；hybrid/local/global 才会走图遍历，"
            "而本项目已 monkey-patch 关掉了实体抽取，图库为空，hybrid 没意义。"
        ),
    )

    st.markdown("#### 召回参数")
    rcol1, rcol2, rcol3 = st.columns(3)
    retrieval_cfg["chunk_top_k"] = rcol1.number_input(
        "chunk_top_k（naive 模式核心）",
        1, 200, int(retrieval_cfg.get("chunk_top_k", 20)),
        help="向量库召回多少个 chunk 进入下一步 rerank。值大召回高、上下文长；值小响应快、可能漏检。",
    )
    retrieval_cfg["top_k"] = rcol2.number_input(
        "top_k（实体/关系层）",
        1, 100, int(retrieval_cfg.get("top_k", 5)),
        help="hybrid/local/global 模式生效；naive 模式无影响。",
    )
    retrieval_cfg["rerank_top_n"] = rcol3.number_input(
        "rerank_top_n",
        1, 50, int(retrieval_cfg.get("rerank_top_n", 10)),
        help="reranker 之后保留多少条进 prompt（启用 reranker 时生效）。",
    )

    st.markdown("#### 阈值参数")
    tcol1, tcol2, tcol3 = st.columns(3)
    retrieval_cfg["cosine_threshold"] = tcol1.slider(
        "cosine_threshold（向量库）",
        0.0, 1.0, float(retrieval_cfg.get("cosine_threshold", 0.2)), step=0.05,
        help="LightRAG 内置：余弦相似度低于此值的 chunk 直接丢弃。0.2 是默认松阈值。",
    )
    retrieval_cfg["score_threshold"] = tcol2.slider(
        "score_threshold（业务层）",
        0.0, 1.0, float(retrieval_cfg.get("score_threshold", 0.6)), step=0.05,
        help="业务侧二次过滤的分数阈值（保留兼容字段）。",
    )
    retrieval_cfg["related_chunk_number"] = tcol3.number_input(
        "related_chunk_number",
        0, 50, int(retrieval_cfg.get("related_chunk_number", 5)),
        help="实体被命中时附带召回多少个相关 chunk（hybrid 模式生效）。",
    )

    st.markdown("#### Token 预算（拼进 prompt 的硬上限）")
    bcol1, bcol2, bcol3 = st.columns(3)
    retrieval_cfg["max_total_tokens"] = bcol1.number_input(
        "max_total_tokens",
        1000, 200000, int(retrieval_cfg.get("max_total_tokens", 30000)), step=1000,
        help="单次检索最终拼进 prompt 的总 token 上限，超过会按优先级裁剪。",
    )
    retrieval_cfg["max_entity_tokens"] = bcol2.number_input(
        "max_entity_tokens",
        100, 50000, int(retrieval_cfg.get("max_entity_tokens", 6000)), step=500,
        help="实体段最大 token（hybrid 生效）。",
    )
    retrieval_cfg["max_relation_tokens"] = bcol3.number_input(
        "max_relation_tokens",
        100, 50000, int(retrieval_cfg.get("max_relation_tokens", 8000)), step=500,
        help="关系段最大 token（hybrid 生效）。",
    )


# ── Tab 5: Chunking（索引时切分策略） ──
with tabs[4]:
    st.warning(
        "⚠️ **改这些参数后必须重建索引**（到 📚 知识库管理页点 force 重新索引）。"
        " 已经存在的 `vdb_chunks.json` 不会被自动重切。"
    )

    ccol1, ccol2 = st.columns(2)
    chunking_cfg["chunk_token_size"] = ccol1.number_input(
        "chunk_token_size（单 chunk 最大 token 数）",
        128, 8192, int(chunking_cfg.get("chunk_token_size", 1200)), step=64,
        help=(
            "切分时每个 chunk 的目标 token 数。常用值：\n"
            "- **800-1200**：精确召回、上下文紧凑（默认 1200）\n"
            "- **1500-2400**：长上下文场景，少量但更完整\n"
            "- **400-600**：极细粒度问答（如 FAQ）"
        ),
    )
    chunking_cfg["chunk_overlap_token_size"] = ccol2.number_input(
        "chunk_overlap_token_size（相邻重叠 token 数）",
        0, 2048, int(chunking_cfg.get("chunk_overlap_token_size", 100)), step=20,
        help=(
            "相邻 chunk 之间的重叠 token，缓解切分边界把语义切断的问题。\n"
            "经验：取 chunk_token_size 的 8-15%。"
        ),
    )
    chunking_cfg["tiktoken_model_name"] = st.text_input(
        "tiktoken_model_name（用于 token 计数的 tokenizer）",
        value=str(chunking_cfg.get("tiktoken_model_name", "gpt-4o-mini")),
        help=(
            "**仅用于切分时数 token**，与你实际用的 LLM 无关。"
            " 默认 `gpt-4o-mini`（cl100k_base，速度快）。中文密集语料可以试 `gpt-4o`。"
        ),
    )

    # 估算性提示
    chunk_size = int(chunking_cfg["chunk_token_size"])
    overlap = int(chunking_cfg["chunk_overlap_token_size"])
    overlap_pct = (overlap / chunk_size * 100) if chunk_size > 0 else 0
    st.caption(
        f"🔢 当前设置：每个 chunk ≈ **{chunk_size}** tokens，"
        f"重叠 **{overlap}** tokens（**{overlap_pct:.1f}%**），"
        f"按经验比例 8-15% 是 {'✅ 健康' if 8 <= overlap_pct <= 15 else '⚠️ 偏离推荐区间'}。"
    )


# ── Tab 6: Multi-Query 改写 ──
with tabs[5]:
    qr_cfg = retrieval_cfg.get("query_rewrite") or {}
    qr_cfg["enabled"] = st.toggle(
        "启用 Multi-Query（query_rewrite.enabled）",
        value=bool(qr_cfg.get("enabled", True)),
        help=(
            "开启后每次检索会先用 LLM 把原 query 改写成 N 条互补 query"
            "（中→英翻译、术语扩展、换视角），并发检索后合并去重。"
        ),
    )
    qr_cfg["llm_model_id"] = st.text_input(
        "改写用 LLM model_id",
        value=str(qr_cfg.get("llm_model_id", "deepseek_v4_flash")),
        help="必须存在于 `config/models.yaml`。建议用非思考模型（响应快、便宜）。",
    )
    q_col1, q_col2, q_col3 = st.columns(3)
    qr_cfg["num_queries"] = q_col1.number_input(
        "num_queries（生成多少条子 query）",
        1, 10, int(qr_cfg.get("num_queries", 3)),
    )
    qr_cfg["temperature"] = q_col2.slider(
        "temperature",
        0.0, 1.5, float(qr_cfg.get("temperature", 0.2)), step=0.05,
    )
    qr_cfg["max_tokens"] = q_col3.number_input(
        "max_tokens",
        64, 4096, int(qr_cfg.get("max_tokens", 400)), step=64,
    )
    qr_cfg["cache_size"] = st.number_input(
        "cache_size（LRU 缓存条数）",
        16, 4096, int(qr_cfg.get("cache_size", 256)), step=16,
        help="同 (domain, query, n) 的改写只跑一次。",
    )
    qr_cfg["max_per_section_chars"] = st.number_input(
        "max_per_section_chars（每路 retrieve 截断长度）",
        128, 20000, int(qr_cfg.get("max_per_section_chars", 2500)), step=128,
    )
    retrieval_cfg["query_rewrite"] = qr_cfg


# ── Tab 7: 原始 YAML 预览 ──
with tabs[6]:
    raw_cfg["rag"] = rag_cfg
    raw_cfg["retrieval"] = retrieval_cfg
    raw_cfg["chunking"] = chunking_cfg
    import yaml as _yaml

    preview_text = _yaml.safe_dump(
        raw_cfg, allow_unicode=True, sort_keys=False, default_flow_style=False
    )
    st.code(preview_text, language="yaml")


# ── 保存 ──
st.divider()
save_col1, save_col2 = st.columns([1, 5])
with save_col1:
    if st.button("💾 保存到磁盘", type="primary"):
        try:
            save_yaml(CONFIG_PATH, raw_cfg)
            st.success(f"✅ 已写入 {CONFIG_PATH}（旧文件备份在 {CONFIG_PATH}.bak）")
            # 清掉缓存的 RAGManager，让下次直查重建
            try:
                st.cache_resource.clear()
            except Exception:
                pass
        except Exception as save_error:
            st.error(f"❌ 保存失败：{save_error}")
with save_col2:
    st.caption(
        "保存后会清空 Streamlit 的 RAGManager 缓存，下次进入「🔍 RAG 直查」会重建。"
    )
