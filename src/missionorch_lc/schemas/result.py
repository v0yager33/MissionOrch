"""COA 生成结果的类型定义。

替代此前各处散落的 Dict[str, Any] 返回值，提供 IDE 自动补全和静态检查。
"""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class _IterationRecord(TypedDict):
    iteration: int
    score: float
    feedback: str
    verdict: str


class _TokenModelUsage(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    call_count: int


class _TokenUsageSummary(TypedDict, total=False):
    by_model: Dict[str, _TokenModelUsage]
    total_tokens: int


class _StageLogEntry(TypedDict, total=False):
    stage: str
    iteration: int
    score: float
    verdict: str
    skipped: bool
    brief_len: int
    coa_len: int
    parse_success: bool
    is_valid: bool
    tools_available: List[str]
    objectives: int
    research_queries: int
    used_knowledge: bool
    reflection_len: int


class COAResult(TypedDict, total=False):
    """run_graph() / COAOrchestrator.generate() 的统一返回类型。"""

    coa_table: str
    final_coa: Dict[str, Any]
    parse_success: bool
    iterations: int
    final_score: float
    best_score: float
    history: List[_IterationRecord]
    validation: Dict[str, Any]
    outputs: Dict[str, Any]
    token_usage: _TokenUsageSummary
    timing: List[Dict[str, Any]]

    # LangGraph 6-Agent 模式专有
    mission_analysis: Dict[str, Any]
    research_brief: str
    stage_log: List[_StageLogEntry]
    rag_sources: List[str]
    rag_enabled: bool

    # 经典模式专有
    config: Dict[str, Any]
