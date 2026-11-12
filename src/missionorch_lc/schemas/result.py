"""COA 生成结果的类型定义。

替代此前各处散落的 Dict[str, Any] 返回值，提供 IDE 自动补全和静态检查。
"""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class _IterationRecordBase(TypedDict):
    iteration: int
    score: float
    feedback: str
    verdict: str


class _IterationRecord(_IterationRecordBase, total=False):
    structure_issues: List[str]
    judge_failed: bool
    is_best: bool


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
    elapsed: float
    truncated: bool
    best_score: float
    stop_reason: str
    expected_outcomes: int
    structure_issues: int
    patched_sections: List[str]
    error: str


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
    stage_timing: Dict[str, float]  # 各节点累计耗时（秒）
    elapsed_seconds: float
    best_iteration: int  # 被采用版本所在的迭代
    stop_reason: str  # quality_met / max_iterations / plateau / planner_error

    # 预期战果：只记录透传，不参与评分；预留给仿真比对与战报
    expected_outcomes: List[Dict[str, Any]]  # 最终方案给出的预期战果（含输入中的条目）
    input_expected_outcomes: List[Dict[str, Any]]  # 输入侧（用户/任务文本/分析）给出的条目

    # 顶层快捷输出（从 outputs 中提取）
    markdown: str

    # LangGraph 6-Agent 模式专有
    mission_analysis: Dict[str, Any]
    research_brief: str
    stage_log: List[_StageLogEntry]
    rag_sources: List[str]
    rag_enabled: bool

    # 运行配置快照（模型 / profile / 循环参数）
    config: Dict[str, Any]
