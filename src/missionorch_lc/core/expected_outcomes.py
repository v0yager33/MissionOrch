"""预期战果（Expected Outcomes）的抽取、归一化与合并。

预期战果只记录、透传，不参与评分和流程决策。它们从三处进入流水线：

1. 调用方显式传入（``run_graph(expected_outcomes=...)`` / ``--expect``），source=user；
2. 任务文本里的"## 预期战果"等段落，由本模块确定性抽取，source=mission_input；
3. Analyst / Planner 输出的结构化条目，source=analyst / planner。

输出侧保证：输入中给出的每一条都会出现在最终方案的预期战果里（``merge_outcomes``），
以便后续把想定脚本拿去仿真后逐条比对实测值、生成战报。
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..schemas.coa import ExpectedOutcome

# 任务文本中可识别的段落标题（不区分大小写）
SECTION_ALIASES: Tuple[str, ...] = (
    "预期战果",
    "预期效果",
    "预计目标",
    "预期目标",
    "期望战果",
    "期望效果",
    "作战指标",
    "效果指标",
    "expected outcomes",
    "expected outcome",
    "moe",
)

_FIELDS = tuple(ExpectedOutcome.model_fields.keys())

# 符号比较符 → 归一化写法
_SYMBOL_CMP = {
    ">=": ">=", "≥": ">=", "＞＝": ">=", "=>": ">=",
    "<=": "<=", "≤": "<=", "＜＝": "<=", "=<": "<=",
    ">": ">", "＞": ">",
    "<": "<", "＜": "<",
    "==": "==", "=": "==", "＝": "==",
    "≈": "≈", "~": "≈", "约": "≈",
}

# 中文比较词 → 归一化写法（长词在前，避免"不少于"被"少于"抢先匹配）
_WORD_CMP: Tuple[Tuple[str, str], ...] = (
    ("不少于", ">="), ("不低于", ">="), ("不小于", ">="), ("至少", ">="), ("以上", ">="),
    ("不超过", "<="), ("不多于", "<="), ("不高于", "<="), ("不大于", "<="), ("至多", "<="),
    ("最多", "<="), ("以内", "<="), ("以下", "<="),
    ("超过", ">"), ("多于", ">"), ("高于", ">"), ("大于", ">"),
    ("少于", "<"), ("低于", "<"), ("小于", "<"),
    ("达到", ">="), ("等于", "=="), ("约为", "≈"),
)

_NUM = r"\d+(?:\.\d+)?"
_RANGE = rf"{_NUM}(?:\s*(?:~|-|–|—|至|到)\s*{_NUM})?"
_UNIT = r"[%‰]|[A-Za-z]+|[\u4e00-\u9fff]{1,3}"

_SYMBOL_RE = re.compile(
    rf"(>=|<=|=>|=<|==|≥|≤|＞＝|＜＝|>|<|＞|＜|=|＝|≈|~|约)\s*({_RANGE})\s*({_UNIT})?"
)
_BARE_NUM_RE = re.compile(rf"({_RANGE})\s*({_UNIT})?")
_RANGE_SPLIT_RE = re.compile(r"\s*(?:~|-|–|—|至|到)\s*")

_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+•·]|\d+[.、)）]|[（(]\d+[)）])\s*(.+)$")
_HEADING_RE = re.compile(r"^\s*#{1,6}\s*(.+?)\s*#*\s*$")
_SPLIT_INLINE_RE = re.compile(r"[；;]\s*")

_SIDE_HINTS = (
    ("RED", ("红方", "敌方", "敌机", "敌", "red")),
    ("BLUE", ("蓝方", "我方", "友方", "己方", "blue")),
)


# ══════════════════════════════════════════════════════════
# 目标值解析
# ══════════════════════════════════════════════════════════


def parse_target(text: str) -> Tuple[str, str, str, int]:
    """从一段文字里解析"比较符 + 数值 + 单位"。

    Returns:
        (comparator, target_value, unit, start)：start 是比较表达式在原文中的起始下标，
        用于把它前面的部分当作衡量指标；解析不到数值时返回 ("", "", "", -1)。
    """
    if not text:
        return "", "", "", -1

    candidates: List[Tuple[int, str, str, str]] = []

    # "30~45 分钟" 里的 ~ 是区间分隔符，不是"约"
    match = next(
        (
            m for m in _SYMBOL_RE.finditer(text)
            if not (m.group(1) == "~" and text[:m.start()].rstrip()[-1:].isdigit())
        ),
        None,
    )
    if match:
        candidates.append(
            (match.start(), _SYMBOL_CMP.get(match.group(1), match.group(1)),
             match.group(2), match.group(3) or "")
        )

    for word, cmp in _WORD_CMP:
        idx = text.find(word)
        if idx < 0:
            continue
        if word in ("以上", "以内", "以下"):
            # 后置比较词："6 架以上" —— 数值在比较词之前
            head = text[:idx]
            nums = list(_BARE_NUM_RE.finditer(head))
            if nums:
                last = nums[-1]
                candidates.append((last.start(), cmp, last.group(1), last.group(2) or ""))
            continue
        tail = _BARE_NUM_RE.search(text, idx + len(word))
        if tail and not text[idx + len(word):tail.start()].strip():
            candidates.append((idx, cmp, tail.group(1), tail.group(2) or ""))

    if candidates:
        start, cmp, value, unit = min(candidates, key=lambda c: c[0])
    else:
        bare = _BARE_NUM_RE.search(text)
        if not bare:
            return "", "", "", -1
        start, cmp, value, unit = bare.start(), "", bare.group(1), bare.group(2) or ""

    value = re.sub(r"\s+", "", value)
    if _RANGE_SPLIT_RE.search(value) and re.match(rf"^{_NUM}\D+{_NUM}$", value):
        low, high = _RANGE_SPLIT_RE.split(value, maxsplit=1)
        value, cmp = f"{low}-{high}", "range"
    return cmp, value, unit.strip(), start


def format_target(outcome: Dict[str, Any]) -> str:
    """把比较符/目标值/单位拼回可读文本，如 ``≥ 6 架``。"""
    value = str(outcome.get("target_value") or "").strip()
    if not value:
        return ""
    cmp = str(outcome.get("comparator") or "").strip()
    unit = str(outcome.get("unit") or "").strip()
    pretty = {">=": "≥", "<=": "≤", "==": "=", "range": ""}.get(cmp, cmp)
    return " ".join(part for part in (pretty, value, unit) if part)


def _infer_side(text: str) -> str:
    lowered = text.lower()
    for side, hints in _SIDE_HINTS:
        if any(h in lowered for h in hints):
            return side
    return ""


def _split_list(value: Any) -> List[str]:
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    if not value:
        return []
    return [p.strip() for p in re.split(r"[;；,，、/]", str(value)) if p.strip()]


# ══════════════════════════════════════════════════════════
# 归一化
# ══════════════════════════════════════════════════════════


def outcome_from_text(text: str, *, source: str = "", index: int = 1) -> Dict[str, Any]:
    """一条自然语言描述 → 结构化预期战果。"""
    text = text.strip().strip("。.").strip()
    cmp, value, unit, start = parse_target(text)
    metric = text[:start].strip(" ：:，,") if start > 0 else ""
    return normalize_outcome(
        {
            "outcome_id": f"EO-{index}",
            "description": text,
            "metric": metric,
            "comparator": cmp,
            "target_value": value,
            "unit": unit,
            "side": _infer_side(metric or text),
        },
        source=source,
        index=index,
    )


_ALIAS_KEYS = {
    "id": "outcome_id", "eo_id": "outcome_id", "战果id": "outcome_id",
    "desc": "description", "outcome": "description", "预期战果": "description",
    "measure": "metric", "衡量指标": "metric",
    "target": "target_value", "value": "target_value", "目标值": "target_value",
    "phase": "phase_id", "关联阶段": "phase_id",
    "effects": "linked_effects", "关联效果": "linked_effects",
    "仿真观测量": "observable", "来源": "source",
}

_SOURCE_ALIASES = {
    "输入": "mission_input", "任务输入": "mission_input", "用户": "user",
    "方案": "planner", "推导": "planner", "分析": "analyst",
}


def normalize_outcome(
    raw: Any, *, source: str = "", index: int = 1
) -> Dict[str, Any]:
    """dict / str / ExpectedOutcome → 字段齐全的 dict（与 ExpectedOutcome 一致）。

    显式给出的 ``source`` 只在条目自身没有来源时生效。
    """
    if isinstance(raw, ExpectedOutcome):
        data: Dict[str, Any] = raw.model_dump()
    elif isinstance(raw, str):
        return outcome_from_text(raw, source=source, index=index)
    elif isinstance(raw, dict):
        data = {}
        for key, val in raw.items():
            norm_key = _ALIAS_KEYS.get(str(key).strip().lower(), str(key).strip())
            data[norm_key] = val
    else:
        data = {"description": str(raw)}

    # "target" 写成 "≥ 6 架" 这类整串时，拆出比较符和单位
    target_text = str(data.get("target_value") or "").strip()
    if target_text and not data.get("comparator"):
        cmp, value, unit, _ = parse_target(target_text)
        if value:
            data["comparator"] = cmp
            data["target_value"] = value
            if not data.get("unit"):
                data["unit"] = unit
    elif not target_text and data.get("description"):
        cmp, value, unit, _ = parse_target(str(data["description"]))
        if value:
            if not data.get("comparator"):
                data["comparator"] = cmp
            data["target_value"] = value
            if not data.get("unit"):
                data["unit"] = unit

    comparator = str(data.get("comparator") or "").strip()
    data["comparator"] = _SYMBOL_CMP.get(comparator, comparator)

    src = str(data.get("source") or "").strip()
    data["source"] = _SOURCE_ALIASES.get(src, src) or source

    data["linked_effects"] = _split_list(data.get("linked_effects"))
    if not data.get("outcome_id"):
        data["outcome_id"] = f"EO-{index}"
    side = str(data.get("side") or "").strip().upper()
    data["side"] = side if side in ("BLUE", "RED", "NEUTRAL") else _infer_side(
        str(data.get("metric") or data.get("description") or "")
    )

    clean = {k: data[k] for k in _FIELDS if k in data}
    for key in ("outcome_id", "description", "metric", "target_value", "unit",
                "phase_id", "priority", "observable"):
        if key in clean and clean[key] is not None:
            clean[key] = str(clean[key]).strip()
    return ExpectedOutcome(**clean).model_dump()


def coerce_outcomes(value: Any, *, source: str = "") -> List[Dict[str, Any]]:
    """把 None / 单条字符串 / 列表（str 或 dict 混合）统一成 dict 列表。"""
    if value is None or value == "":
        return []
    if isinstance(value, (str, dict, ExpectedOutcome)):
        value = [value]
    out: List[Dict[str, Any]] = []
    for item in value:
        if isinstance(item, str):
            parts = [p for p in _SPLIT_INLINE_RE.split(item) if p.strip()]
        else:
            parts = [item]
        for part in parts:
            out.append(normalize_outcome(part, source=source, index=len(out) + 1))
    return out


# ══════════════════════════════════════════════════════════
# 从任务文本中抽取
# ══════════════════════════════════════════════════════════


def _is_alias(title: str) -> bool:
    lowered = re.sub(r"^[一二三四五六七八九十\d]+[、.．)\s]*", "", title.strip()).lower()
    lowered = lowered.strip(" ：:")
    return any(lowered.startswith(alias) for alias in SECTION_ALIASES)


def _parse_table(lines: Sequence[str]) -> List[Dict[str, str]]:
    rows = [ln.strip() for ln in lines if ln.strip().startswith("|")]
    if len(rows) < 2:
        return []
    headers = [h.strip() for h in rows[0].strip("|").split("|")]
    records = []
    for row in rows[1:]:
        cells = [c.strip() for c in row.strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
            continue
        records.append(dict(zip(headers, cells)))
    return records


_TABLE_HEADER_KEYS = {
    "战果id": "outcome_id", "id": "outcome_id",
    "预期战果": "description", "描述": "description", "战果": "description",
    "衡量指标": "metric", "指标": "metric",
    "目标值": "target_value", "目标": "target_value",
    "关联阶段": "phase_id", "阶段": "phase_id",
    "关联效果": "linked_effects",
    "仿真观测量": "observable", "观测量": "observable",
    "优先级": "priority", "来源": "source",
}


def table_rows_to_outcomes(
    rows: Iterable[Dict[str, str]], *, source: str = ""
) -> List[Dict[str, Any]]:
    """Markdown 表格行（表头为中文列名）→ 预期战果列表。"""
    out: List[Dict[str, Any]] = []
    for row in rows:
        mapped: Dict[str, Any] = {}
        for header, cell in row.items():
            key = header.strip().lower()
            for alias, field in _TABLE_HEADER_KEYS.items():
                if alias in key:
                    mapped.setdefault(field, cell)
                    break
        if not any(mapped.get(k) for k in ("description", "metric", "target_value")):
            continue
        out.append(normalize_outcome(mapped, source=source, index=len(out) + 1))
    return out


def extract_from_mission(text: str) -> List[Dict[str, Any]]:
    """从任务描述里抽取"## 预期战果"（及同义）段落中的条目。

    支持三种写法：
    - Markdown 标题 + 列表 / 表格；
    - 单行"预期战果：A；B；C"；
    - 单行标题"预期战果："后跟列表。
    """
    if not text:
        return []
    lines = text.splitlines()
    items: List[str] = []
    table_lines: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        heading = _HEADING_RE.match(line)
        inline = None
        if heading and _is_alias(heading.group(1)):
            pass
        else:
            m = re.match(r"^\s*(?:[-*]\s*)?\**([^：:]{2,20}?)\**\s*[：:]\s*(.*)$", line)
            if m and _is_alias(m.group(1)):
                inline = m.group(2).strip()
            else:
                i += 1
                continue

        if inline:
            items.extend(p for p in _SPLIT_INLINE_RE.split(inline) if p.strip())
        i += 1
        # 收集段落正文，直到下一个标题或同级的"xx："行
        while i < len(lines):
            body = lines[i]
            if _HEADING_RE.match(body):
                break
            if inline is not None and body.strip() and not (
                _LIST_ITEM_RE.match(body) or body.strip().startswith("|")
            ):
                break
            if body.strip().startswith("|"):
                table_lines.append(body)
            else:
                item = _LIST_ITEM_RE.match(body)
                if item:
                    items.extend(
                        p for p in _SPLIT_INLINE_RE.split(item.group(1)) if p.strip()
                    )
                elif body.strip() and inline is None:
                    items.extend(p for p in _SPLIT_INLINE_RE.split(body) if p.strip())
            i += 1

    outcomes = table_rows_to_outcomes(_parse_table(table_lines), source="mission_input")
    for item in items:
        outcomes.append(
            outcome_from_text(item, source="mission_input", index=len(outcomes) + 1)
        )
    return _renumber(outcomes)


# ══════════════════════════════════════════════════════════
# 合并
# ══════════════════════════════════════════════════════════


def _norm_text(text: Any) -> str:
    return re.sub(r"[\s\W_]+", "", str(text or "")).lower()


def _bigrams(text: str) -> set:
    return {text[i:i + 2] for i in range(len(text) - 1)} if len(text) > 1 else {text}


def _similar(a: Any, b: Any, threshold: float = 0.5) -> bool:
    na, nb = _norm_text(a), _norm_text(b)
    if not na or not nb:
        return False
    if na in nb or nb in na:
        return True
    ba, bb = _bigrams(na), _bigrams(nb)
    return len(ba & bb) / max(1, len(ba | bb)) >= threshold


def _same_outcome(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    if _similar(a.get("description"), b.get("description")):
        return True
    same_metric = _similar(a.get("metric"), b.get("metric"), threshold=0.6)
    same_target = _norm_text(a.get("target_value")) == _norm_text(b.get("target_value"))
    if same_metric and same_target:
        return True
    return (
        bool(a.get("outcome_id"))
        and a.get("outcome_id") == b.get("outcome_id")
        and (same_metric or same_target)
    )


def _renumber(outcomes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set = set()
    for idx, outcome in enumerate(outcomes, start=1):
        oid = outcome.get("outcome_id") or f"EO-{idx}"
        if oid in seen:
            n = idx
            while f"EO-{n}" in seen:
                n += 1
            oid = f"EO-{n}"
        outcome["outcome_id"] = oid
        seen.add(oid)
    return outcomes


def merge_outcomes(
    primary: Optional[Iterable[Any]],
    required: Optional[Iterable[Any]] = None,
    *,
    keep_primary_source: bool = False,
) -> List[Dict[str, Any]]:
    """以 ``primary`` 为主合并预期战果，保证 ``required`` 中每一条都在结果里。

    - 匹配到的条目：用 required 补齐 primary 的空字段；来源默认以 required 为准
      （``keep_primary_source=True`` 时保留 primary 的来源）；
    - 未匹配到的 required 条目：原样追加；
    - 最后按出现顺序去重编号（EO-1、EO-2 ...）。
    """
    merged = [normalize_outcome(o, index=i) for i, o in enumerate(primary or [], start=1)]
    for req in required or []:
        req = normalize_outcome(req, index=len(merged) + 1)
        match = next((m for m in merged if _same_outcome(m, req)), None)
        if match is None:
            merged.append(req)
            continue
        for key, val in req.items():
            if key in ("actual_value", "status"):
                continue
            if val and not match.get(key):
                match[key] = val
        if req.get("source") and not (keep_primary_source and match.get("source")):
            match["source"] = req["source"]
    return _renumber(merged)


# ══════════════════════════════════════════════════════════
# 渲染
# ══════════════════════════════════════════════════════════

_SOURCE_LABELS = {
    "user": "用户", "mission_input": "输入", "analyst": "分析", "planner": "方案",
}
_STATUS_LABELS = {
    "PENDING": "待仿真", "MET": "达成", "PARTIAL": "部分达成", "NOT_MET": "未达成",
}


def _cell(value: Any) -> str:
    text = ", ".join(value) if isinstance(value, list) else str(value or "")
    return text.replace("|", "/").replace("\n", " ").strip() or "—"


def outcomes_to_markdown(
    outcomes: Sequence[Dict[str, Any]], *, with_actual: bool = False
) -> str:
    """渲染为 Markdown 表格；``with_actual`` 时追加"仿真实测 / 状态"两列（预留）。"""
    if not outcomes:
        return ""
    headers = ["战果ID", "预期战果", "衡量指标", "目标值", "关联阶段", "关联效果", "仿真观测量", "来源"]
    if with_actual:
        headers += ["仿真实测", "状态"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join([":---"] * len(headers)) + "|"]
    for o in outcomes:
        row = [
            o.get("outcome_id"),
            o.get("description"),
            o.get("metric"),
            format_target(o),
            o.get("phase_id"),
            o.get("linked_effects"),
            o.get("observable"),
            _SOURCE_LABELS.get(o.get("source", ""), o.get("source")),
        ]
        if with_actual:
            row += [
                o.get("actual_value"),
                _STATUS_LABELS.get(o.get("status", ""), o.get("status")),
            ]
        lines.append("| " + " | ".join(_cell(c) for c in row) + " |")
    return "\n".join(lines)


def outcomes_to_prompt(outcomes: Sequence[Dict[str, Any]]) -> str:
    """给 Planner 的输入片段：要求原样保留这些条目。"""
    if not outcomes:
        return ""
    return (
        "## 输入给定的预期战果（必须原样出现在第八节，来源填\"输入\"）\n\n"
        + outcomes_to_markdown(outcomes)
    )
