"""离线轨迹管线 CLI；只有显式 synthesize 才调用模型。"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from contextlib import contextmanager
from pathlib import Path

from .core import digest, get_recorder, private_dir, scope
from .processing import DataError, correction, export, load_rows, process, read_events, write_rows


def synthesis_prompt():
    if __package__.startswith("missionorch_lc"):
        from missionorch_lc.core.prompt_loader import load_prompt_text
        return load_prompt_text(Path(__file__).parents[1] / "prompts" / "trajectory_synthesis.txt")
    from nanobot.utils.prompt_templates import render_template
    return render_template("agent/trajectory_synthesis.txt")


async def model_client(model=None, preset=None, max_output_tokens=4096):
    if __package__.startswith("missionorch_lc"):
        from missionorch_lc.agents.base import load_agents_config
        from missionorch_lc.core.model_router import ModelRouter
        selected = model or load_agents_config(profile=preset)["agents"]["planner"]["model_id"]
        llm = ModelRouter.get(selected)

        async def complete(messages):
            response = await llm.ainvoke(messages, max_tokens=max_output_tokens)
            return response.content
        return selected, complete, None
    from nanobot.config.loader import load_config
    from nanobot.providers.factory import make_provider
    config = load_config()
    selected = model or config.resolve_preset(preset).model
    provider = make_provider(config, preset_name=preset, model=selected)

    async def complete(messages):
        response = await provider.chat_with_retry(messages=messages, model=selected, max_tokens=max_output_tokens)
        if response.finish_reason == "error":
            raise DataError("合成模型调用失败；详见本地轨迹")
        return response.content
    return selected, complete, getattr(provider, "aclose", None)


@contextmanager
def synthesis_lock(workdir):
    """本地单工作目录互斥；进程退出自动解锁，保留稳定锁文件。"""
    private_dir(workdir)
    fd = os.open(Path(workdir) / ".synthesis.lock", os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        if os.name == "nt":
            import msvcrt
            if os.fstat(fd).st_size == 0:
                os.write(fd, b" ")
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        raise DataError("该工作目录已有合成进程；本次未调用模型") from None
    try:
        yield
    finally:
        os.close(fd)


async def synthesize(workdir, complete, model, *, limit=10, concurrency=2, max_input_chars=0):
    if limit < 0 or concurrency < 1 or max_input_chars < 0:
        raise DataError("无效的合成限额")
    with synthesis_lock(Path(workdir).resolve()):
        return await _synthesize( workdir, complete, model, limit=limit, concurrency=concurrency,
            max_input_chars=max_input_chars)


async def _synthesize(workdir, complete, model, *, limit, concurrency, max_input_chars):
    workdir = Path(workdir).resolve()
    sources = load_rows(workdir / "original.jsonl")
    prompt = synthesis_prompt()
    checkpoints = load_rows(workdir / "checkpoint.jsonl")
    done = {row["key"] for row in checkpoints}
    proposals = load_rows(workdir / "synthetic.jsonl")
    # 结果先落盘、checkpoint 尚未落盘的崩溃窗口也不能重复付费合成。
    done.update(row["checkpoint_key"] for row in proposals if row.get("checkpoint_key"))
    semaphore = asyncio.Semaphore(concurrency)
    selected = []
    for source in sources:
        key = digest({"source": source, "model": model, "prompt": prompt})
        if key not in done and source["status"] != "verified":
            selected.append((source, key))
            if len(selected) >= limit:
                break
    async def one(source, key):
        async with semaphore:
            payload = {k: v for k, v in source.items() if k != "events"}
            # 保留有关错误、真实修改与验证的证据，禁止模型把这些观察伪造为执行。
            payload["events"] = source.get("events", [])
            related = [ s for s in sources if s[
                "id"] != source[
                "id"]
                and s[
                "lineage"] == source[
                "lineage"]
                and any( d.get(
                "sample_id") == source[
                "id"]
                for d in s.get(
                "depends_on",
                [ ]))]
            payload[
                "related_revisions"] = [
                { k: v for k, v in s.items( ) if k not in {
                "events",
                "actual_requests"}}
                for s in related]
            related_ids = [event for s in related for event in s["source_event_ids"]]
            encoded = json.dumps(payload, ensure_ascii=False)
            try:
                if max_input_chars and len(encoded) > max_input_chars:
                    raise DataError("候选超过输入限额；未截断原始内容")
                with scope( trace_id=source.get(
                    "trace_id"),
                    session_id=source.get(
                    "lineage"),
                    agent="trajectory_synthesis",
                    source_sample_id=source[
                    "id"]):
                    text = await complete([{"role": "system", "content": prompt}, {"role": "user", "content": encoded}])
                if isinstance(text, list):
                    text = "".join(part.get("text", "") for part in text if isinstance(part, dict))
                text = str(text).strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[1].rsplit("```", 1)[0]
                proposal = get_recorder().redact(json.loads(text))
                related_hashes = {event: s["artifact_hash"] for s in related for event in s["source_event_ids"]}
                result = correction({**source, "related_event_ids": related_ids,
                                     "related_artifact_hashes": related_hashes}, proposal, model)
                result["checkpoint_key"] = key
                proposals.append(result)
                record = {
                    "key": key,
                    "source_id": source[
                    "id"],
                    "model": model,
                    "status": result[
                    "status"],
                    "result_id": result[
                    "id"]}
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                record = {
                    "key": key,
                    "source_id": source[
                    "id"],
                    "model": model,
                    "status": "failed",
                    "error_type": type(
                    exc).__name__}
            checkpoints.append(record)
            # 单 event loop 中串行原子持久化，完成一个即保存一个，不重写原始采集。
            write_rows(workdir / "synthetic.jsonl", proposals)
            write_rows(workdir / "checkpoint.jsonl", checkpoints)
    if limit:
        await asyncio.gather(*(one(source, key) for source, key in selected))
    return {"selected": len(selected) if limit else 0, "checkpointed": len(checkpoints), "synthetic": len(proposals)}


def parser():
    result = argparse.ArgumentParser(description="本地轨迹采集与蒸馏数据管线；不重放代码/工具")
    commands = result.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="检查原始采集事件与残缺统计")
    inspect.add_argument("inputs", nargs="+")
    processing = commands.add_parser("process", help="生成 original/corrected/recovery 关联视图")
    processing.add_argument("inputs", nargs="+")
    processing.add_argument("--workdir", required=True)
    synthesis = commands.add_parser("synthesize", help="显式调用配置模型生成校正候选")
    synthesis.add_argument("--workdir", required=True)
    synthesis.add_argument("--model")
    synthesis.add_argument("--preset")
    synthesis.add_argument("--limit", type=int, default=10)
    synthesis.add_argument("--concurrency", type=int, default=2)
    synthesis.add_argument("--max-output-tokens", type=int, default=4096)
    synthesis.add_argument("--max-input-chars", type=int, default=0)
    exporting = commands.add_parser("export", help="默认仅导出 verified；恢复仅监督修复回答")
    exporting.add_argument("--workdir", required=True)
    exporting.add_argument("--outdir", required=True)
    exporting.add_argument("--status", action="append", choices=["verified", "candidate"])
    exporting.add_argument("--validation-fraction", type=float, default=0.1)
    return result


def main(argv=None):
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    try:
        if args.command == "inspect":
            events, stats = read_events(args.inputs)
            for event in events:
                key = "event_" + event.get("event", "unknown")
                stats[key] += 1
            output = dict(stats)
        elif args.command == "process":
            output = process(args.inputs, args.workdir)
        elif args.command == "export":
            if not 0 <= args.validation_fraction <= 1:
                raise DataError("validation-fraction 必须位于 [0, 1]")
            output = export(args.workdir, args.outdir, tuple(args.status or ["verified"]), args.validation_fraction)
        else:
            if args.limit < 0 or args.concurrency < 1 or args.max_output_tokens < 1 or args.max_input_chars < 0:
                raise DataError("合成限额不能为负；并发和输出 token 上限必须为正")
            async def execute():
                if not args.limit:
                    return {"selected": 0}
                model, complete, close = await model_client(args.model, args.preset, args.max_output_tokens)
                try:
                    return await synthesize( args.workdir, complete, model, limit=args.limit,
                        concurrency=args.concurrency, max_input_chars=args.max_input_chars)
                finally:
                    if close:
                        await close()
            output = asyncio.run(execute())
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0
    except (DataError, OSError, ValueError) as exc:
        argument_parser.error(str(exc))
