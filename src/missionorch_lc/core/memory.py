"""COA 记忆模块 —— 将每次运行的输入/输出持久化为 Markdown，供 RAG 检索。

每次成功的 COA 生成都会保存为一份结构化 Markdown 文件：
- 包含完整的任务输入
- 包含系统生成的 COA 矩阵
- 包含评估分数、迭代轨迹等元数据
- 存储在 knowledge_base/memory/ 目录下

这使得系统可以通过 RAG 检索历史规划经验，实现"越用越强"的效果。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from .expected_outcomes import outcomes_to_markdown

logger = logging.getLogger(__name__)

MEMORY_DIR = Path(__file__).parents[3] / "knowledge_base" / "memory"


def save_run_memory(
    mission_input: str,
    result: Dict[str, Any],
    *,
    scenario_name: str = "",
    memory_dir: Path | None = None,
) -> Path | None:
    """将一次 COA 生成的完整输入/输出保存为 Markdown 文件。

    Args:
        mission_input: 用户输入的任务描述
        result: run_graph() 返回的完整结果字典
        scenario_name: 可选的场景标识名（用于文件命名）
        memory_dir: 存储目录，默认 knowledge_base/memory/

    Returns:
        保存的文件路径，失败返回 None
    """
    if memory_dir is None:
        memory_dir = MEMORY_DIR

    memory_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = scenario_name or "coa_run"
    filename = f"{slug}_{timestamp}.md"
    filepath = memory_dir / filename

    score = result.get("final_score", 0)
    iterations = result.get("iterations", 0)
    parse_success = result.get("parse_success", False)
    coa_table = result.get("coa_table", "")
    history = result.get("history", [])
    token_usage = result.get("token_usage", {})
    validation = result.get("validation", {})
    analysis = result.get("mission_analysis", {})

    md_parts = [
        f"# COA 规划记录: {slug}",
        f"\n> 生成时间: {timestamp}  ",
        f"> 最终得分: {score}  ",
        f"> 迭代次数: {iterations}  ",
        f"> 解析成功: {'是' if parse_success else '否'}  ",
        f"> Token总量: {token_usage.get('total_tokens', 0):,}",
        "",
        "---",
        "",
        "## 任务输入",
        "",
        mission_input,
        "",
        "---",
        "",
    ]

    if analysis:
        md_parts.extend([
            "## 任务分析结果",
            "",
            f"- 意图: {analysis.get('mission_intent', '')}",
            f"- 目标数: {len(analysis.get('objectives', []))}",
        ])
        objs = analysis.get("objectives", [])
        for obj in objs:
            md_parts.append(
                f"  - [{obj.get('id')}] ({obj.get('priority')}) "
                f"{obj.get('description', '')}"
            )
        entities = analysis.get("key_entities", {})
        md_parts.append(f"- 蓝方力量: {', '.join(entities.get('blue_forces', []))}")
        md_parts.append(f"- 红方力量: {', '.join(entities.get('red_forces', []))}")
        md_parts.append(f"- 关键区域: {', '.join(entities.get('areas', []))}")
        md_parts.extend(["", "---", ""])

    outcomes = result.get("expected_outcomes") or []
    if outcomes:
        md_parts.extend([
            "## 预期战果",
            "",
            "> 只记录，不参与评分；仿真后回填实测值用于比对。",
            "",
            outcomes_to_markdown(outcomes, with_actual=True),
            "",
            "---",
            "",
        ])

    md_parts.extend([
        "## COA 矩阵输出",
        "",
        coa_table,
        "",
        "---",
        "",
    ])

    if history:
        md_parts.extend([
            "## 迭代历史",
            "",
            "| 迭代 | 得分 | 判定 | 反馈摘要 |",
            "|:---|:---|:---|:---|",
        ])
        for h in history:
            feedback_short = h.get("feedback", "")[:80]
            md_parts.append(
                f"| {h['iteration']} | {h['score']} | {h.get('verdict', '')} | "
                f"{feedback_short} |"
            )
        md_parts.extend(["", "---", ""])

    if validation:
        is_valid = validation.get("is_valid", False)
        issues = validation.get("issues_found", [])
        md_parts.extend([
            "## 验证结果",
            "",
            f"- 通过: {'是' if is_valid else '否'}",
            f"- 问题数: {len(issues)}",
        ])
        for issue in issues[:10]:
            md_parts.append(f"  - {issue}")
        md_parts.extend([""])

    final_coa = result.get("final_coa", {})
    if final_coa and parse_success:
        md_parts.extend([
            "",
            "## 结构化 COA 数据（JSON）",
            "",
            "```json",
            json.dumps(final_coa, ensure_ascii=False, indent=2)[:5000],
            "```",
        ])

    content = "\n".join(md_parts)

    try:
        filepath.write_text(content, encoding="utf-8")
        logger.info(f"[Memory] 已保存规划记录: {filepath} ({len(content)} chars)")
        return filepath
    except Exception as e:
        logger.error(f"[Memory] 保存失败: {e}")
        return None


def list_memories(memory_dir: Path | None = None) -> list[Path]:
    """列出所有已保存的记忆文件。"""
    if memory_dir is None:
        memory_dir = MEMORY_DIR
    if not memory_dir.exists():
        return []
    return sorted(memory_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
