#!/usr/bin/env python3
"""MissionOrch-LC CLI 入口。

支持两种编排模式：
    python main.py                  # 经典 Orchestrator（默认）
    python main.py --graph          # LangGraph StateGraph 版
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

# 允许直接 `python main.py` 运行
sys.path.insert(0, str(Path(__file__).parent / "src"))

from missionorch_lc.core.log_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


# ── CLI 参数 ──


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MissionOrch-LC — 基于 LangChain 的 COA 编排系统"
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        help="使用 LangGraph StateGraph 版本编排（默认使用经典循环版）",
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
    """友好地打印 COA 生成结果。"""
    print("\n" + "=" * 70)
    print("【COA 矩阵方案】")
    print("=" * 70)
    print(result.get("coa_table", "(empty)"))
    print("=" * 70)

    print("\n### 生成信息")
    print(f"- **迭代次数**: {result.get('iterations', 0)}")
    print(f"- **最终得分**: {result.get('final_score', 0)}")
    print(f"- **最佳得分**: {result.get('best_score', 0)}")
    print(f"- **矩阵解析**: {'✅ 成功' if result.get('parse_success') else '❌ 失败'}")

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
) -> Dict[str, Any]:
    """运行 LangGraph 版 Orchestrator。"""
    from missionorch_lc.orchestrator_graph import run_graph

    kwargs: Dict[str, Any] = {}
    if max_iter is not None:
        kwargs["max_iterations"] = max_iter
    if threshold is not None:
        kwargs["quality_threshold"] = threshold

    print(
        f"📋 LangGraph 模式 | max_iter={kwargs.get('max_iterations', 3)}, "
        f"threshold={kwargs.get('quality_threshold', 8.0)}"
    )
    print("⏳ 正在生成 COA 方案...\n")
    return await run_graph(mission, **kwargs)


# ── 交互式输入 ──


async def async_input(prompt: str) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: input(prompt))


async def interactive_loop(args: argparse.Namespace) -> int:
    """交互式循环：反复读取任务描述并生成 COA。"""
    runner = run_langgraph if args.graph else run_classic
    mode_name = "LangGraph" if args.graph else "经典"

    print(f"\n🚀 {mode_name}模式已就绪。输入任务描述开始生成（输入 quit 退出）:\n")

    while True:
        mission_input = (await async_input("任务描述: ")).strip()
        if mission_input.lower() in ("quit", "exit", "q"):
            print("👋 退出系统")
            break
        if not mission_input:
            print("⚠️  任务描述不能为空，请重新输入\n")
            continue

        try:
            result = await runner(
                mission_input,
                max_iter=args.max_iter,
                threshold=args.threshold,
            )
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