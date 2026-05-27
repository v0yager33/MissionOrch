"""本地 Qwen3 Embedding / Reranker 封装。

把 `/chatgpt_nas/dukaixuan.dkx/models/Qwen3-Embedding-0.6B` 与
`Qwen3-Reranker-0.6B` 封装为 LightRAG 可消费的回调：
- `build_local_embedding_func(model_path, ...) -> EmbeddingFunc`
- `build_local_rerank_func(model_path, ...) -> async callable`

依赖：transformers / torch（PyTorch）。

设计要点：
1. 模型在第一次调用时懒加载并缓存到模块级（同一进程多次 build 共享底层权重）
2. embedding 走 last-token pooling + L2 normalize（Qwen3-Embedding 推荐做法）
3. reranker 走 generative 风格（Qwen3-Reranker 推荐做法）：拼成 yes/no
   判别 prompt，取 yes token 概率作为相关性分数
4. 全部封装为 async，兼容 LightRAG 的 `await embedding_func(texts)` 调用约定
"""

from __future__ import annotations

import asyncio
import logging
import math
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any, Callable, List, Optional

logger = logging.getLogger(__name__)

# ── 延迟导入：避免在没装 torch 的环境里 import 时炸 ──
_torch = None
_AutoModel = None
_AutoTokenizer = None
_AutoModelForCausalLM = None


def _ensure_transformers() -> None:
    """惰性加载 torch / transformers，给出友好错误。"""
    global _torch, _AutoModel, _AutoTokenizer, _AutoModelForCausalLM
    if _torch is not None:
        return
    try:
        import torch  # type: ignore
        from transformers import (  # type: ignore
            AutoModel,
            AutoModelForCausalLM,
            AutoTokenizer,
        )
    except ImportError as import_error:
        raise ImportError(
            "本地 Qwen3 embedding / reranker 需要 torch + transformers，"
            "请先 `pip install torch transformers` 或参考 docs/RAG_GUIDE.md"
        ) from import_error
    _torch = torch
    _AutoModel = AutoModel
    _AutoTokenizer = AutoTokenizer
    _AutoModelForCausalLM = AutoModelForCausalLM


# ── 进程级模型缓存 ──
_model_cache: dict = {}
_cache_lock = Lock()


def _last_token_pool(last_hidden_states: Any, attention_mask: Any) -> Any:
    """Qwen3-Embedding 的官方推荐 pooling：最后一个有效 token。"""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        _torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


# ─────────────────────────────────────────────
# Embedding
# ─────────────────────────────────────────────


class _Qwen3Embedder:
    """单例化的 Qwen3 embedding 推理器。"""

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        max_length: int = 8192,
        batch_size: int = 8,
        instruction: Optional[str] = None,
    ) -> None:
        _ensure_transformers()
        self.model_path = model_path
        self.max_length = max_length
        self.batch_size = batch_size
        # Qwen3-Embedding 推荐：在 query 前面加一段任务描述
        self.instruction = instruction or (
            "Given a web search query, retrieve relevant passages that answer the query"
        )

        if device == "auto":
            device = "cuda" if _torch.cuda.is_available() else "cpu"
        self.device = device

        logger.info(f"Loading Qwen3 Embedding from {model_path} on {device} ...")
        self.tokenizer = _AutoTokenizer.from_pretrained(
            model_path, padding_side="left", trust_remote_code=True
        )
        self.model = _AutoModel.from_pretrained(
            model_path, trust_remote_code=True
        ).to(device).eval()
        self.embedding_dim = self.model.config.hidden_size
        logger.info(
            f"Qwen3 Embedding ready: dim={self.embedding_dim}, max_length={max_length}"
        )

    def _format_query(self, text: str, is_query: bool) -> str:
        if is_query:
            return f"Instruct: {self.instruction}\nQuery: {text}"
        return text

    def _encode_batch(self, texts: List[str], is_query: bool) -> Any:
        formatted = [self._format_query(t, is_query) for t in texts]
        batch = self.tokenizer(
            formatted,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        with _torch.no_grad():
            outputs = self.model(**batch)
        embeddings = _last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
        # L2 normalize → 余弦距离 = 内积
        embeddings = _torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

    def encode(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        if not texts:
            return []
        results: List[List[float]] = []
        for offset in range(0, len(texts), self.batch_size):
            chunk = texts[offset: offset + self.batch_size]
            embeddings = self._encode_batch(chunk, is_query=is_query)
            results.extend(embeddings.cpu().tolist())
        return results


def _get_or_build_embedder(model_path: str, **kwargs: Any) -> _Qwen3Embedder:
    """进程级缓存：相同 model_path 复用同一份权重。"""
    key = (str(Path(model_path).resolve()), tuple(sorted(kwargs.items())))
    with _cache_lock:
        cached = _model_cache.get(("emb", key))
        if cached is not None:
            return cached
        embedder = _Qwen3Embedder(model_path, **kwargs)
        _model_cache[("emb", key)] = embedder
        return embedder


def build_local_embedding_func(
    model_path: str,
    *,
    device: str = "auto",
    max_length: int = 8192,
    batch_size: int = 8,
    instruction: Optional[str] = None,
):
    """构建 LightRAG 兼容的 `EmbeddingFunc`（异步可调用）。

    返回的对象签名为 `async def(texts: List[str]) -> np.ndarray`。

    Args:
        model_path: Qwen3-Embedding 模型目录（如 ``/.../Qwen3-Embedding-0.6B``）
        device: ``"auto"`` / ``"cpu"`` / ``"cuda"`` / ``"cuda:0"``
        max_length: 输入截断长度
        batch_size: 每个 batch 的最大文本数
        instruction: query 前缀指令，None 则用通用检索指令
    """
    # lightrag 1.4+ 把 EmbeddingFunc 从 lightrag.utils 迁移到 lightrag.types
    EmbeddingFunc = None
    try:
        from lightrag.utils import EmbeddingFunc as _EmbeddingFunc  # type: ignore

        EmbeddingFunc = _EmbeddingFunc
    except ImportError:
        try:
            from lightrag.types import EmbeddingFunc as _EmbeddingFunc  # type: ignore

            EmbeddingFunc = _EmbeddingFunc
        except ImportError as ie:
            raise ImportError(
                "lightrag 未安装。请 `pip install lightrag-hku` 或参考 docs/RAG_GUIDE.md"
            ) from ie

    try:
        import numpy as np  # type: ignore
    except ImportError as ie:
        raise ImportError("需要 numpy，请 `pip install numpy`") from ie

    embedder = _get_or_build_embedder(
        model_path,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
        instruction=instruction,
    )

    async def _embed(texts: List[str]) -> Any:
        # transformers 是同步的 → 用线程池避免阻塞 event loop
        loop = asyncio.get_running_loop()
        # 默认 LightRAG 走 passage 编码（is_query=False）；
        # query 编码会被 LightRAG 自己包装时携带不同 instruction
        vectors = await loop.run_in_executor(
            None, lambda: embedder.encode(list(texts), is_query=False)
        )
        return np.array(vectors, dtype=np.float32)

    return EmbeddingFunc(
        embedding_dim=embedder.embedding_dim,
        max_token_size=max_length,
        func=_embed,
    )


# ─────────────────────────────────────────────
# Reranker
# ─────────────────────────────────────────────


class _Qwen3Reranker:
    """Qwen3-Reranker：用 generative yes/no 概率给候选打分。"""

    PROMPT_TEMPLATE = (
        "<|im_start|>system\nJudge whether the Document meets the requirements based "
        "on the Query and the Instruct provided. Note that the answer can only be "
        "\"yes\" or \"no\".<|im_end|>\n"
        "<|im_start|>user\n<Instruct>: {instruction}\n<Query>: {query}\n"
        "<Document>: {document}<|im_end|>\n<|im_start|>assistant\n\n\n"
    )

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        max_length: int = 8192,
        instruction: Optional[str] = None,
    ) -> None:
        _ensure_transformers()
        self.model_path = model_path
        self.max_length = max_length
        self.instruction = instruction or (
            "Given a web search query, retrieve relevant passages that answer the query"
        )

        if device == "auto":
            device = "cuda" if _torch.cuda.is_available() else "cpu"
        self.device = device

        logger.info(f"Loading Qwen3 Reranker from {model_path} on {device} ...")
        self.tokenizer = _AutoTokenizer.from_pretrained(
            model_path, padding_side="left", trust_remote_code=True
        )
        self.model = _AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True
        ).to(device).eval()
        self.token_yes_id = self.tokenizer("yes", add_special_tokens=False).input_ids[0]
        self.token_no_id = self.tokenizer("no", add_special_tokens=False).input_ids[0]
        logger.info("Qwen3 Reranker ready")

    def _build_prompt(self, query: str, document: str) -> str:
        return self.PROMPT_TEMPLATE.format(
            instruction=self.instruction,
            query=query,
            document=document,
        )

    def score(self, query: str, documents: List[str]) -> List[float]:
        """返回每个 document 的相关性分数（0~1，越大越相关）。"""
        if not documents:
            return []
        prompts = [self._build_prompt(query, d) for d in documents]
        scores: List[float] = []
        # 一次只跑一条，简单可靠；如要并行可拼 batch
        with _torch.no_grad():
            for prompt in prompts:
                inputs = self.tokenizer(
                    prompt,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                outputs = self.model(**inputs)
                # 取最后一个位置上 yes / no 的 logit，做 softmax
                logits = outputs.logits[0, -1, :]
                yes_logit = logits[self.token_yes_id].item()
                no_logit = logits[self.token_no_id].item()
                # 数值稳定的 softmax(2-class)
                m = max(yes_logit, no_logit)
                yes_p = math.exp(yes_logit - m)
                no_p = math.exp(no_logit - m)
                scores.append(yes_p / (yes_p + no_p))
        return scores


def _get_or_build_reranker(model_path: str, **kwargs: Any) -> _Qwen3Reranker:
    key = (str(Path(model_path).resolve()), tuple(sorted(kwargs.items())))
    with _cache_lock:
        cached = _model_cache.get(("rerank", key))
        if cached is not None:
            return cached
        reranker = _Qwen3Reranker(model_path, **kwargs)
        _model_cache[("rerank", key)] = reranker
        return reranker


def build_local_rerank_func(
    model_path: str,
    *,
    device: str = "auto",
    max_length: int = 8192,
    instruction: Optional[str] = None,
) -> Callable:
    """构建 LightRAG 兼容的异步 rerank 函数。

    LightRAG 的 `rerank_model_func` 约定：
        async def rerank(query: str, documents: List[Dict|str], top_n: int) -> List[Dict]
    返回带 ``relevance_score`` 字段的有序列表。
    """
    reranker = _get_or_build_reranker(
        model_path, device=device, max_length=max_length, instruction=instruction
    )

    def _doc_text(doc: Any) -> str:
        if isinstance(doc, str):
            return doc
        if isinstance(doc, dict):
            return doc.get("content") or doc.get("text") or str(doc)
        return str(doc)

    async def _rerank(
        query: str,
        documents: List[Any],
        top_n: int = 10,
        **kwargs: Any,
    ) -> List[dict]:
        if not documents:
            return []
        loop = asyncio.get_running_loop()
        texts = [_doc_text(d) for d in documents]
        scores = await loop.run_in_executor(
            None, lambda: reranker.score(query, texts)
        )
        ranked = sorted(
            (
                {**(d if isinstance(d, dict) else {"content": d}), "relevance_score": float(s)}
                for d, s in zip(documents, scores, strict=True)
            ),
            key=lambda x: x["relevance_score"],
            reverse=True,
        )
        return ranked[:top_n]

    return _rerank


@lru_cache(maxsize=1)
def is_local_rag_available() -> bool:
    """探测一下 torch + transformers 是否已就位。"""
    try:
        _ensure_transformers()
        return True
    except ImportError:
        return False
