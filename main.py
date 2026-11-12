#!/usr/bin/env python3
"""MissionOrch-LC CLI 入口。

默认走 6-Agent LangGraph 流水线：
    python main.py                  # 交互模式
    python main.py --mission "..."  # 非交互
    python main.py --legacy         # 退回旧版 4-Agent 经典 Orchestrator
    python main.py --no-rag         # 关闭 RAG（Researcher 节点降级）
    python main.py --profile fast   # 运行档位：fast / balanced（默认）/ quality
    python main.py --mission-file m.md --expect "击落敌机≥4架" --expect "我方损失≤1架"
    python main.py --help           # 查看帮助

预期战果（--expect / 任务文本中的"## 预期战果"段落）只透传到输出，不参与评分，
预留给后续仿真比对与战报生成。

环境变量：
    DEEPSEEK_API_KEY                       —— DeepSeek 密钥（必填）
    MISSIONORCH_PROFILE                    —— 默认运行档位
    LANGSMITH_API_KEY / LANGSMITH_PROJECT  —— 自动启用 LangSmith tracing
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

try:
    import missionorch_lc  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent / "src"))

from missionorch_lc.core.log_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


# ── CLI 参数 ──


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MissionOrch-LC — 基于 LangChain + LangGraph 的 6-Agent COA 编排系统"
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="使用经典 4-Agent 循环（默认走 LangGraph 6-Agent 流水线）",
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="禁用 RAG（Researcher 节点降级，不读知识库）",
    )
    parser.add_argument(
        "--mission",
        type=str,
        default="",
        help="直接传入任务描述（非交互模式）",
    )
    parser.add_argument(
        "--mission-file",
        type=str,
        default="",
        help="从文件读取任务描述（UTF-8，非交互模式）",
    )
    parser.add_argument(
        "--expect",
        action="append",
        default=[],
        metavar="OUTCOME",
        help="预期战果，可重复，如 --expect \"击落敌机≥4架\"（只透传到输出，不参与评分）",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="运行档位（agents.yaml 的 profiles：fast / balanced / quality），"
             "默认读 MISSIONORCH_PROFILE 或 workflow.default_profile",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=None,
        help="覆盖最大迭代次数（默认读 agents.yaml）",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="覆盖质量阈值（默认读 agents.yaml）",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="将完整结果输出到指定 JSON 文件",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default="",
        help="将结构化 COA 方案输出到指定 Markdown 文件（推荐格式，可被 ScenClaw 直接消费）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="统一输出路径，自动按扩展名选择格式（.md / .json / .yaml），"
             "如 --output result.md 等价于 --output-md result.md",
    )
    return parser.parse_args()


# ── 结果展示 ──

_STOP_REASON_LABELS = {
    "quality_met": "达到质量阈值",
    "max_iterations": "达到迭代上限",
    "plateau": "分数进入平台期，提前结束",
    "planner_error": "修订调用失败，保留已有最优版本",
}


def print_result(result: Dict[str, Any]) -> None:
    """友好地打印 COA 生成结果（含 6-Agent 流水线的 analyst / researcher 阶段）。"""
    # ── Analyst 阶段 ──
    analysis = result.get("mission_analysis") or {}
    if analysis:
        print("\n" + "=" * 70)
        print("【1/6 Analyst — 任务分析】")
        print("=" * 70)
        print(f"意图: {analysis.get('mission_intent', '')}")
        objs = analysis.get("objectives", [])
        if objs:
            print(f"目标 ({len(objs)}):")
            for o in objs:
                print(
                    f"  - [{o.get('id')}] ({o.get('priority')}) "
                    f"{o.get('description', '')}"
                )
        constraints = analysis.get("constraints", [])
        if constraints:
            print(f"约束: {'; '.join(constraints)}")
        queries = analysis.get("research_queries", [])
        if queries:
            print(f"建议研究问题 ({len(queries)}):")
            for q in queries:
                print(f"  · {q}")

    # ── Researcher 阶段 ──
    brief = result.get("research_brief", "")
    if brief:
        print("\n" + "=" * 70)
        print("【2/6 Researcher — 研究简报】")
        print("=" * 70)
        # 简报太长就截断，避免淹没终端
        print(brief if len(brief) < 2000 else brief[:2000] + "\n...(已截断)")

    # ── COA 矩阵 ──
    print("\n" + "=" * 70)
    print("【3/6 Planner — COA 矩阵方案】")
    print("=" * 70)
    print(result.get("coa_table", "(empty)"))
    print("=" * 70)

    print("\n### 生成信息")
    print(f"- **迭代次数**: {result.get('iterations', 0)}")
    print(f"- **采用版本得分**: {result.get('final_score', 0)}"
          + (f"（第 {result['best_iteration']} 轮）" if result.get("best_iteration") else ""))
    if result.get("stop_reason"):
        print(f"- **停止原因**: {_STOP_REASON_LABELS.get(result['stop_reason'], result['stop_reason'])}")
    if result.get("elapsed_seconds") is not None:
        print(f"- **总耗时**: {result['elapsed_seconds']:.1f}s")
    profile = (result.get("config") or {}).get("profile")
    if profile:
        print(f"- **运行档位**: {profile}")
    print(f"- **矩阵解析**: {'✅ 成功' if result.get('parse_success') else '❌ 失败'}")
    if "rag_enabled" in result:
        rag_state = "启用" if result["rag_enabled"] else "未启用"
        sources = result.get("rag_sources") or []
        print(
            f"- **RAG**: {rag_state} ({len(sources)} sources: "
            f"{', '.join(sources) or '—'})"
        )

    if result.get("parse_success"):
        final_coa = result.get("final_coa", {})
        print(f"- **阶段数量**: {len(final_coa.get('phases', []))}")
        print(f"- **作战单元**: {len(final_coa.get('units', []))}")
        print(f"- **矩阵单元格**: {len(final_coa.get('matrix', []))}")
        print(f"- **效果链**: {len(final_coa.get('effects_chain', []))}")

    # 预期战果（只透传，不参与评分）
    outcomes = result.get("expected_outcomes") or []
    if outcomes:
        from missionorch_lc.core.expected_outcomes import format_target

        print(f"\n### 预期战果 ({len(outcomes)}，不参与评分，待仿真回填)")
        for o in outcomes:
            target = format_target(o)
            parts = [o.get("description", "")]
            if o.get("metric") or target:
                parts.append(f"{o.get('metric', '')} {target}".strip())
            if o.get("phase_id"):
                parts.append(f"阶段 {o['phase_id']}")
            print(f"  - [{o.get('outcome_id')}] " + " | ".join(p for p in parts if p)
                  + f"  （来源: {o.get('source') or '—'}）")

    # 迭代历史
    history = result.get("history", [])
    if history:
        print("\n### 迭代历史")
        for entry in history:
            print(
                f"  迭代 {entry['iteration']}: "
                f"得分 {entry['score']:.1f} | 判定 {entry.get('verdict', 'N/A')}"
                + (f" | 结构问题 {len(entry['structure_issues'])}" if "structure_issues" in entry else "")
                + (" | ★采用" if entry.get("is_best") and entry["iteration"] == result.get("best_iteration") else "")
            )

    # Token 用量
    token_usage = result.get("token_usage", {})
    if token_usage:
        total = token_usage.get("total_tokens", 0)
        print(f"\n### Token 用量 (总计: {total:,})")
        by_model = token_usage.get("by_model", {})
        for model_name, usage in by_model.items():
            print(
                f"  {model_name}: prompt={usage['prompt_tokens']:,} "
                f"completion={usage['completion_tokens']:,} "
                f"total={usage['total_tokens']:,} "
                f"({usage['call_count']} calls)"
            )

    # 阶段耗时（LangGraph 模式按节点累计；经典模式按顶层 chain）
    stage_timing = result.get("stage_timing") or {}
    timing = result.get("timing", [])
    if stage_timing:
        print("\n### 阶段耗时")
        for name, elapsed in stage_timing.items():
            print(f"  {name}: {elapsed:.2f}s")
    elif timing:
        print("\n### 阶段耗时")
        for stage in timing:
            elapsed = stage.get("elapsed")
            if elapsed is not None:
                tags = ", ".join(stage.get("tags", []))
                name = stage.get("run_name", "")
                label = name or tags or "unknown"
                print(f"  {label}: {elapsed:.2f}s")

    # 验证
    validation = result.get("validation", {})
    if validation:
        is_valid = validation.get("is_valid", False)
        issues = validation.get("issues_found", [])
        print(
            f"\n### 验证结果: {'✅ 通过' if is_valid else '⚠️ 未通过'}"
            f" ({len(issues)} issues)"
        )
        if issues:
            for issue in issues[:5]:
                print(f"  - {issue}")


# ── 结果持久化 ──


def save_outputs(result: Dict[str, Any], args: argparse.Namespace) -> None:
    """根据 CLI 参数将结果保存到指定文件（支持多格式并行输出）。

    支持三种触发方式：
    - ``--output-json result.json`` : 完整结果 JSON
    - ``--output-md   result.md``   : 结构化 COA 方案 Markdown
    - ``--output      result.md``   : 按扩展名自动选择格式
      (.md -> Markdown, .json -> JSON, .yaml/.yml -> YAML)
    """
    import yaml as _yaml

    saved: list[str] = []

    # ── --output-json（保留向后兼容）──
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        saved.append(args.output_json)

    # ── --output-md ──
    if args.output_md:
        md_content = result.get("markdown", "") or result.get("coa_table", "")
        Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_md, "w", encoding="utf-8") as f:
            f.write(md_content)
        saved.append(args.output_md)

    # ── --output（按扩展名路由）──
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ext = out_path.suffix.lower()

        if ext == ".md":
            md_content = result.get("markdown", "") or result.get("coa_table", "")
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(md_content)
        elif ext == ".json":
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        elif ext in (".yaml", ".yml"):
            # YAML 优先输出结构化 COA，如不可用则输出完整 result
            coa_data = result.get("final_coa", {}) or result
            with open(args.output, "w", encoding="utf-8") as f:
                _yaml.dump(coa_data, f, default_flow_style=False, allow_unicode=True)
        else:
            # 默认 JSON
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        saved.append(args.output)

    for path in saved:
        print(f"\n💾 结果已保存到 {path}")


# ── 运行器 ──


async def run_classic(
    mission: str,
    max_iter: int | None = None,
    threshold: float | None = None,
    *,
    expected_outcomes: list[str] | None = None,
    profile: str | None = None,
) -> Dict[str, Any]:
    """运行经典 Orchestrator。"""
    from missionorch_lc.orchestrator import COAOrchestrator

    orchestrator = COAOrchestrator(profile=profile)
    if max_iter is not None:
        orchestrator.max_iter = max_iter
    if threshold is not None:
        orchestrator.threshold = threshold

    print(
        f"📋 经典模式 | max_iter={orchestrator.max_iter}, "
        f"threshold={orchestrator.threshold}"
    )
    print("⏳ 正在生成 COA 方案...\n")
    return await orchestrator.generate(mission, expected_outcomes=expected_outcomes)


async def run_langgraph(
    mission: str,
    max_iter: int | None = None,
    threshold: float | None = None,
    use_rag: bool = True,
    *,
    expected_outcomes: list[str] | None = None,
    profile: str | None = None,
) -> Dict[str, Any]:
    """运行 LangGraph 6-Agent Orchestrator。"""
    from missionorch_lc.agents.base import load_agents_config
    from missionorch_lc.orchestrator_graph import run_graph

    workflow = load_agents_config(profile=profile).get("workflow") or {}
    shown_iter = max_iter if max_iter is not None else workflow.get("max_iterations", 3)
    shown_threshold = threshold if threshold is not None else workflow.get("quality_threshold", 8.0)
    flow = "reflector ⇄ " if workflow.get("use_reflector") else ""
    print(
        f"📋 LangGraph 6-Agent 模式 | max_iter={shown_iter}, "
        f"threshold={shown_threshold}, rag={'on' if use_rag else 'off'}, "
        f"revision={workflow.get('revision_mode', 'full')}"
    )
    print(f"⏳ 流程: analyst → researcher → planner ⇄ {flow}judge → finalize\n")
    return await run_graph(
        mission,
        use_rag=use_rag,
        max_iterations=max_iter,
        quality_threshold=threshold,
        expected_outcomes=expected_outcomes or None,
        profile=profile,
    )


# ── 调度 ──


async def _dispatch(mission: str, args: argparse.Namespace) -> Dict[str, Any]:
    """根据 CLI 参数选择运行器。"""
    if args.legacy:
        return await run_classic(
            mission, max_iter=args.max_iter, threshold=args.threshold,
            expected_outcomes=args.expect, profile=args.profile,
        )
    return await run_langgraph(
        mission,
        max_iter=args.max_iter,
        threshold=args.threshold,
        use_rag=not args.no_rag,
        expected_outcomes=args.expect,
        profile=args.profile,
    )


# ── 交互式输入 ──


async def async_input(prompt: str) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: input(prompt))


async def interactive_loop(args: argparse.Namespace) -> int:
    """交互式循环：反复读取任务描述并生成 COA。"""
    mode_name = "经典 4-Agent" if args.legacy else "LangGraph 6-Agent"
    print(f"\n🚀 {mode_name} 模式已就绪。输入任务描述开始生成（输入 quit 退出）:\n")

    while True:
        mission_input = (await async_input("任务描述: ")).strip()
        if mission_input.lower() in ("quit", "exit", "q"):
            print("👋 退出系统")
            break
        if not mission_input:
            print("⚠️  任务描述不能为空，请重新输入\n")
            continue

        try:
            result = await _dispatch(mission_input, args)
            print_result(result)
            save_outputs(result, args)
        except Exception as generate_error:
            logger.error(f"COA 生成失败: {generate_error}", exc_info=True)
            print(f"\n❌ 生成失败: {generate_error}\n")

    return 0


# ── 主入口 ──


async def main() -> int:
    args = parse_args()

    print("MissionOrch-LC — 基于 LangChain 的 COA 编排系统")
    print("=" * 60)

    # 尝试激活 LangSmith tracing
    try:
        from missionorch_lc.core.settings import AppSettings

        settings = AppSettings()
        settings.tracing.activate()
    except Exception:
        pass  # settings 加载失败不影响主流程

    try:
        if args.mission_file:
            args.mission = Path(args.mission_file).read_text(encoding="utf-8").strip()
        if args.mission:
            # 非交互模式
            result = await _dispatch(args.mission, args)
            print_result(result)
            save_outputs(result, args)
            return 0
        else:
            # 交互模式
            return await interactive_loop(args)

    except KeyboardInterrupt:
        print("\n\n👋 程序被用户中断")
        return 0
    except Exception as system_error:
        logger.error(f"系统错误: {system_error}", exc_info=True)
        print(f"\n❌ 系统错误: {system_error}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)