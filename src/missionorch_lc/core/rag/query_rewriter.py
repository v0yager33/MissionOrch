"""Query Rewriter —— Multi-Query 查询改写器。

为什么要做 query rewriting？
  - 我们的 doctrines / historical 语料是英文，但 Analyst 产出的 research_queries
    往往是中文。纯向量检索在跨语言时几乎必然失败（embedding 空间不对齐）。
  - 用户/Analyst 的 query 往往是抽象问题（"空地协同的关键要点"），而文档
    里是具体陈述（"Joint Forces Air Component Commander shall coordinate..."）。
    同一 query 从不同视角展开可以大幅提升召回。

P0 实现（Multi-Query）：
  - 让 LLM 一次性把原始 query 改写成 N=3 条：
      1. 原文（保留用户意图）
      2. 中→英翻译版（跨语言对齐）
      3. 术语扩展版（领域同义词 / 展开缩略语）
  - 3 条 query 并发调用 `rag_manager.retrieve`
  - 结果合并：按文本前缀去重、拼接成带标题的单段文本返回

显式不做（YAGNI）：
  - HyDE：朴素 RAG 下 LLM "假答案" 如果错就带偏检索
  - Step-Back：COA 场景是领域内具体问题，过度抽象反而降精度
  - Chunk-level RRF：aquery 返回的是合成文本不是 chunk 列表，没得 RRF

缓存：
  - (source, query) → list[rewritten_query] 的 LRU 缓存（默认 256 条）
  - Researcher ReAct 循环里可能对同一 query 多次调用，避免重复花 LLM 钱
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import OrderedDict
from typing import Awaitable, Callable, List, Optional, Tuple

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


# ── Prompt 构造 ──
# system prompt 里 query 条数和规则会随 num_queries 动态变化（避免硬编码）
def _build_system_prompt(num_queries: int) -> str:
    """按 num_queries 动态生成 system prompt，避免硬编码导致条数不一致。"""
    return (
        f"你是军事情报检索专家。给定一个检索需求，你需要生成 {num_queries} 条"
        f"互补的英文检索 query，覆盖不同视角，提高向量检索的召回率。\n\n"
        "规则：\n"
        f"1. {num_queries} 条 query 都必须是英文（主要语料是英文军事文档）。\n"
        "2. 第 1 条：直译 / 字面翻译原始需求（保留意图）。\n"
        "3. 第 2 条：用领域术语 / 缩略语重写"
        "（展开或收敛，如 \"空地协同\" → \"close air support (CAS) coordination\"）。\n"
        "4. 其余若干条：换视角展开（更具体的操作细节 / 更宏观的原则层面 / 相关的概念族）。\n"
        "5. 每条 query 15 词以内，陈述性关键词串，不要疑问句。\n"
        "6. 严格只输出 JSON："
        "{\"queries\": [\"...\", ...]}，不要任何其它文字、不要代码块包裹。"
    )


# few-shot：user 提问 + assistant 输出，让 LLM 学到"我应该这样回"而不是"用户在教我"
_FEW_SHOT_USER = "领域：military doctrines\n原始需求：空地协同打击的关键要点"
_FEW_SHOT_ASSISTANT = (
    '{"queries": ['
    '"key points of air-ground coordination for strike operations", '
    '"close air support (CAS) procedures and joint terminal attack controller responsibilities", '
    '"command and control principles for joint fires across air and ground components"'
    ']}'
)


# (domain, query, num_queries) → 改写结果
CacheKey = Tuple[str, str, int]


class _LRUCache:
    """极简 OrderedDict LRU —— asyncio 事件循环单线程访问，无需加锁。"""

    def __init__(self, max_size: int = 256) -> None:
        self._store: "OrderedDict[CacheKey, List[str]]" = OrderedDict()
        self._max_size = max_size

    def get(self, key: CacheKey) -> Optional[List[str]]:
        value = self._store.get(key)
        if value is not None:
            self._store.move_to_end(key)
        return value

    def put(self, key: CacheKey, value: List[str]) -> None:
        self._store[key] = value
        self._store.move_to_end(key)
        if len(self._store) > self._max_size:
            self._store.popitem(last=False)


# 调用方给 ``retrieve_with_rewrite`` 的检索回调：async (query) -> str
RetrieveFn = Callable[[str], Awaitable[str]]


class MultiQueryRewriter:
    """基于 LLM 的多路 query 改写器。

    调用方约定：
      - ``rewrite(query, domain)`` 返回 List[str]（改写失败时至少返回 ``[query]``）
      - ``retrieve_with_rewrite(retrieve_fn, query, domain)`` 并发调用 retrieve_fn，
        合并去重后返回拼接文本
    """

    def __init__(
        self,
        *,
        llm_model_id: str = "deepseek_v4_flash",
        num_queries: int = 3,
        cache_size: int = 256,
        enabled: bool = True,
        temperature: float = 0.2,
        max_tokens: int = 400,
    ) -> None:
        self.llm_model_id = llm_model_id
        self.num_queries = max(1, num_queries)
        self.enabled = enabled
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._cache = _LRUCache(max_size=cache_size)
        # 懒加载 ChatModel：避免 import 期就去读 models.yaml
        self._raw_model: Optional[BaseChatModel] = None
        # system prompt 按 num_queries 预先烘焙一次，避免每次改写都拼字符串
        self._system_prompt = _build_system_prompt(self.num_queries)
        logger.info(
            "MultiQueryRewriter initialized: model=%s, num_queries=%d, enabled=%s",
            llm_model_id,
            self.num_queries,
            enabled,
        )

    # ── 内部：懒加载 ChatModel ──
    def _get_model(self) -> BaseChatModel:
        if self._raw_model is None:
            # 延迟导入，避免循环引用（core.rag → core.model_router）
            from ..model_router import ModelRouter

            self._raw_model = ModelRouter.get_raw(self.llm_model_id)
        return self._raw_model

    # ── 对外 API ──
    async def rewrite(self, query: str, domain: str = "military") -> List[str]:
        """把 query 改写成 N 条互补 query。失败时回退到 ``[query]``。"""
        if not self.enabled or not query or not query.strip():
            return [query] if query else []

        stripped = query.strip()
        cache_key: CacheKey = (domain, stripped, self.num_queries)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            rewritten = await self._call_llm(stripped, domain)
        except Exception as rewrite_error:
            logger.warning(
                "MultiQueryRewriter: 改写失败，降级为原 query: %s", rewrite_error
            )
            rewritten = [stripped]

        # 统一出口：去空、去完全重复；永远保证至少有 [query]
        unique: List[str] = []
        seen: set = set()
        for candidate in rewritten:
            cleaned = (candidate or "").strip()
            if cleaned and cleaned not in seen:
                unique.append(cleaned)
                seen.add(cleaned)
        if not unique:
            unique = [stripped]

        self._cache.put(cache_key, unique)
        return unique

    async def _call_llm(self, query: str, domain: str) -> List[str]:
        """调 LLM 一次性产出 N 条 query，解析 JSON。"""
        model = self._get_model()
        user_prompt = (
            f"领域：{domain}\n"
            f"原始需求：{query}\n\n"
            f"请严格按 JSON 格式输出 {self.num_queries} 条。"
        )

        # 正确的 few-shot：User → Assistant → User，让 LLM 看到
        # "上一条 assistant 消息是这么回的"，而不是 "用户在教我格式"。
        messages = [
            SystemMessage(content=self._system_prompt),
            HumanMessage(content=_FEW_SHOT_USER),
            AIMessage(content=_FEW_SHOT_ASSISTANT),
            HumanMessage(content=user_prompt),
        ]

        # 低温、短 max_tokens，避免啰嗦。``bind`` 返回 Runnable，不改原模型单例。
        bound = model.bind(temperature=self.temperature, max_tokens=self.max_tokens)
        response = await bound.ainvoke(messages)
        raw_text = response.content if hasattr(response, "content") else str(response)
        return self._parse_queries(str(raw_text))

    @staticmethod
    def _parse_queries(raw_text: str) -> List[str]:
        """从 LLM 输出里提取 queries 数组。

        健壮性：
        1. 直接 ``json.loads`` 整串
        2. 失败则用 ``json.JSONDecoder().raw_decode`` 从第一个 ``{`` 起做流式解析
           —— 比 ``re.search(r"\\{.*\\}")`` 正确处理嵌套 / 多余文字
        3. 拿到 dict 后要求 ``queries`` 必须是 list，且每项可转成非空 str
        4. 全部失败返回 ``[]``，由调用方决定是否回退到原 query
        """
        if not raw_text:
            return []

        text = raw_text.strip()
        # 去掉可能的 ```json ... ``` 代码块包裹
        if text.startswith("```"):
            # 找到第一行换行后的内容、去掉结尾的 ```
            newline_idx = text.find("\n")
            if newline_idx >= 0:
                text = text[newline_idx + 1 :]
            if text.endswith("```"):
                text = text[: -3]
            text = text.strip()

        parsed: Optional[dict] = None
        # Pass 1：整串 JSON
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            pass

        # Pass 2：从第一个 '{' 开始 raw_decode，兼容 LLM 在 JSON 前/后带解释文字
        if parsed is None:
            brace_idx = text.find("{")
            if brace_idx >= 0:
                decoder = json.JSONDecoder()
                try:
                    parsed_obj, _end = decoder.raw_decode(text[brace_idx:])
                    if isinstance(parsed_obj, dict):
                        parsed = parsed_obj
                except json.JSONDecodeError:
                    parsed = None

        if not isinstance(parsed, dict):
            return []

        raw_queries = parsed.get("queries")
        if not isinstance(raw_queries, list):
            return []

        result: List[str] = []
        for item in raw_queries:
            if isinstance(item, str):
                cleaned = item.strip()
            elif item is not None:
                cleaned = str(item).strip()
            else:
                cleaned = ""
            if cleaned:
                result.append(cleaned)
        return result

    async def retrieve_with_rewrite(
        self,
        retrieve_fn: RetrieveFn,
        query: str,
        domain: str = "military",
        *,
        max_per_section_chars: int = 2500,
    ) -> str:
        """并发调用 retrieve_fn，合并结果为带小标题的拼接文本。

        Args:
            retrieve_fn: 形如 ``async def f(query: str) -> str``
                （通常是闭包：``lambda q: rag_manager.retrieve(q, source=..., mode=...)``）
            query: 原始 query（改写前）
            domain: 领域提示词
            max_per_section_chars: 每一路结果截断长度，避免最终 prompt 过长
        """
        rewritten = await self.rewrite(query, domain=domain)
        if not rewritten:
            return ""

        # 并发检索：return_exceptions=True 让任一失败不影响其它路
        results = await asyncio.gather(
            *(retrieve_fn(sub_query) for sub_query in rewritten),
            return_exceptions=True,
        )

        sections: List[str] = []
        seen_prefixes: set = set()
        for sub_query, result in zip(rewritten, results, strict=True):
            if isinstance(result, BaseException):
                logger.warning(
                    "MultiQueryRewriter: retrieve 子查询失败 (%s): %s",
                    sub_query,
                    result,
                )
                continue

            text = (result or "").strip() if isinstance(result, str) else ""
            if not text:
                continue

            # 去重：用文本前 120 字符作指纹
            # （aquery 的输出可能在多路 query 下返回高度相似的合成答案）
            fingerprint = text[:120]
            if fingerprint in seen_prefixes:
                continue
            seen_prefixes.add(fingerprint)

            if len(text) > max_per_section_chars:
                overflow = len(text) - max_per_section_chars
                text = text[:max_per_section_chars] + f"\n...(truncated, {overflow} more chars)"
            sections.append(f"### Sub-query: {sub_query}\n{text}")

        if not sections:
            return ""

        header = (
            f"> Retrieved via multi-query rewrite "
            f"({len(sections)}/{len(rewritten)} sub-queries hit)\n\n"
        )
        return header + "\n\n---\n\n".join(sections)
