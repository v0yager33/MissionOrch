#!/usr/bin/env python3
"""MissionOrch-LC CLI 入口。

默认走 6-Agent LangGraph 流水线：
    python main.py                  # 交互模式
    python main.py --mission "..."  # 非交互
    python main.py --legacy         # 退回旧版 4-Agent 经典 Orchestrator
    python main.py --no-rag         # 关闭 RAG（Researcher 节点降级）
    python main.py --help           # 查看帮助

环境变量：
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
    return parser.parse_args()


# ── 结果展示 ──


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
    print(f"- **最终得分**: {result.get('final_score', 0)}")
    print(f"- **最佳得分**: {result.get('best_score', 0)}")
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

    # 迭代历史
    history = result.get("history", [])
    if history:
        print("\n### 迭代历史")
        for entry in history:
            print(
                f"  迭代 {entry['iteration']}: "
                f"得分 {entry['score']:.1f} | 判定 {entry.get('verdict', 'N/A')}"
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

    # 阶段耗时
    timing = result.get("timing", [])
    if timing:
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


# ── 运行器 ──


async def run_classic(
    mission: str,
    max_iter: int | None = None,
    threshold: float | None = None,
) -> Dict[str, Any]:
    """运行经典 Orchestrator。"""
    from missionorch_lc.orchestrator import COAOrchestrator

    orchestrator = COAOrchestrator()
    if max_iter is not None:
        orchestrator.max_iter = max_iter
    if threshold is not None:
        orchestrator.threshold = threshold

    print(
        f"📋 经典模式 | max_iter={orchestrator.max_iter}, "
        f"threshold={orchestrator.threshold}"
    )
    print("⏳ 正在生成 COA 方案...\n")
    return await orchestrator.generate(mission)


async def run_langgraph(
    mission: str,
    max_iter: int | None = None,
    threshold: float | None = None,
    use_rag: bool = True,
) -> Dict[str, Any]:
    """运行 LangGraph 6-Agent Orchestrator。"""
    from missionorch_lc.orchestrator_graph import run_graph

    kwargs: Dict[str, Any] = {"use_rag": use_rag}
    if max_iter is not None:
        kwargs["max_iterations"] = max_iter
    if threshold is not None:
        kwargs["quality_threshold"] = threshold

    print(
        f"📋 LangGraph 6-Agent 模式 | max_iter={kwargs.get('max_iterations', 3)}, "
        f"threshold={kwargs.get('quality_threshold', 8.0)}, "
        f"rag={'on' if use_rag else 'off'}"
    )
    print("⏳ 流程: analyst → researcher → planner ⇄ judge ⇄ reflector → finalize\n")
    return await run_graph(mission, **kwargs)


# ── 调度 ──


async def _dispatch(mission: str, args: argparse.Namespace) -> Dict[str, Any]:
    """根据 CLI 参数选择运行器。"""
    if args.legacy:
        return await run_classic(
            mission, max_iter=args.max_iter, threshold=args.threshold
        )
    return await run_langgraph(
        mission,
        max_iter=args.max_iter,
        threshold=args.threshold,
        use_rag=not args.no_rag,
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

            if args.output_json:
                with open(args.output_json, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2, default=str)
                print(f"\n💾 结果已保存到 {args.output_json}")
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
        if args.mission:
            # 非交互模式
            result = await _dispatch(args.mission, args)
            print_result(result)

            if args.output_json:
                with open(args.output_json, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2, default=str)
                print(f"\n💾 结果已保存到 {args.output_json}")
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