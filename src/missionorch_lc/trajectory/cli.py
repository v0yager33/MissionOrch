"""轨迹修复与一键蒸馏管线；默认全量模型修复，--offline 禁用模型调用。"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

from .core import digest, get_recorder, private_dir, scope
from .processing import (
    DataError,
    correction,
    export,
    import_evidence,
    input_roots,
    load_rows,
    overlaps,
    process,
    read_events,
    repair,
    validate_workdir,
    write_rows,
)


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


async def synthesize(workdir, complete, model, *, limit=None, concurrency=2, max_input_chars=0):
    if (limit is not None and limit < 0) or concurrency < 1 or max_input_chars < 0:
        raise DataError("无效的合成限额")
    with synthesis_lock(Path(workdir).resolve()):
        return await _synthesize( workdir, complete, model, limit=limit, concurrency=concurrency,
            max_input_chars=max_input_chars)


def synthesis_sources(workdir, repair_only):
    workdir = Path(workdir)
    sources = load_rows(workdir / "original.jsonl")
    repair_ids = {row["source_id"] for row in load_rows(workdir / "repair-queue.jsonl")}
    repair_ids.update(row["repair_of"] for row in load_rows(workdir / "repairs.jsonl"))
    return [source for source in sources if (not repair_only or source["id"] in repair_ids)
            and source["status"] != "verified" and source["messages"][:-1]
            and not any(event.get("collection_errors") for event in source.get("events", []))]


async def _synthesize(workdir, complete, model, *, limit, concurrency, max_input_chars, repair_only=False):
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
    for source in synthesis_sources(workdir, repair_only) if limit != 0 else []:
        key = digest({"source_id": source["id"], "messages": source["messages"],
                      "source_event_ids": source["source_event_ids"], "model": model, "prompt": prompt})
        legacy_key = digest({"source": source, "model": model, "prompt": prompt})
        if key not in done and legacy_key not in done:
            selected.append((source, key))
            if limit is not None and len(selected) >= limit:
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
                with scope(trace_id="synthesis:" + key, session_id="synthesis:" + source["lineage"],
                    source_trace_id=source.get("trace_id"), agent="trajectory_synthesis",
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
    pending = iter(selected)
    async def worker():
        for source, key in pending:
            await one(source, key)
    await asyncio.gather(*(worker() for _ in range(min(concurrency, len(selected)))))
    return {"selected": len(selected), "checkpointed": len(checkpoints), "synthetic": len(proposals)}


async def configured_synthesis(args):
    if args.limit == 0 or not synthesis_sources(args.workdir, args.command != "synthesize"):
        return {"selected": 0}
    model, complete, close = await model_client(args.model, args.preset, args.max_output_tokens)
    try:
        return await _synthesize(args.workdir, complete, model, limit=args.limit, concurrency=args.concurrency,
                                 max_input_chars=args.max_input_chars, repair_only=args.command != "synthesize")
    finally:
        if close:
            await close()


def check_build_paths(inputs, workdir, outdir):
    if overlaps(workdir, outdir):
        raise DataError("处理目录与数据包目录必须独立")
    for root in input_roots(inputs):
        if overlaps(workdir, root) or overlaps(outdir, root):
            raise DataError("处理/数据包目录不能与原始采集目录重叠")


def build_bundle(workdir, outdir, validation_fraction, stages):
    outdir = Path(outdir).resolve()
    verified = export(workdir, outdir, ("verified",), validation_fraction)
    candidates = export(workdir, outdir / "candidates", ("candidate",), validation_fraction)
    queue = load_rows(Path(workdir) / "repair-queue.jsonl")
    repaired_ids = {row.get("repair_of") for name in ("repairs.jsonl", "synthetic.jsonl")
                    for row in load_rows(Path(workdir) / name) if row["status"] in {"candidate", "verified"}}
    pending = [row for row in queue if row["source_id"] not in repaired_ids]
    write_rows(outdir / "pending-repairs.jsonl", pending)
    requests = [{"sample_id": row["id"], "artifact_hash": row["artifact_hash"], "input_hash": row["input_hash"],
                 "source": None, "validation_scope": None, "passed": None}
                for name in ("corrected.jsonl", "recovery.jsonl", "repairs.jsonl", "synthetic.jsonl")
                for row in load_rows(Path(workdir) / name) if row["status"] == "candidate"]
    write_rows(outdir / "pending-verification.jsonl", requests)
    manifest = {"format": "trajectory-distillation/v1", "stages": stages,
                "ready": bool(verified["dataset_train"] + verified["dataset_validation"]),
                "verified": verified, "candidates": candidates, "pending_repairs": len(pending),
                "pending_verification": len(requests),
                "training_files": ["train.messages.jsonl", "validation.messages.jsonl"],
                "completion_files": ["train.completion.jsonl", "validation.completion.jsonl"],
                "candidate_directory": "candidates", "supervision": "last_assistant_only",
                "split_group": "original_task_lineage", "executes_trajectory_code": False,
                "media": "media_ref 相对于所在数据集目录解析；资源保存在 media/ 中"}
    names = [f"{split}.{form}.jsonl" for split in ("train", "validation") for form in ("messages", "completion")]
    names += [f"{kind}.{split}.{form}.jsonl" for kind in ("success", "recovery")
              for split in ("train", "validation") for form in ("messages", "completion")]
    names += ["statistics.jsonl", "provenance.jsonl", "media-index.jsonl"]
    files = []
    for prefix in (Path(), Path("candidates")):
        assets = [row["media_ref"] for row in load_rows(outdir / prefix / "media-index.jsonl")]
        for name in names + assets:
            path = outdir / prefix / name
            files.append({"path": str(prefix / name), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest["files"] = files
    write_rows(outdir / "manifest.jsonl", [manifest])
    return manifest


async def build(inputs, workdir, outdir, *, complete=None, model=None, limit=None, concurrency=2,
                max_input_chars=0, evidence=(), validation_fraction=0.1, offline=False,
                preset=None, max_output_tokens=4096):
    """默认按项目配置全量修复；offline 禁用模型，complete 可注入离线测试客户端。"""
    workdir, outdir = Path(workdir).resolve(), Path(outdir).resolve()
    check_build_paths(inputs, workdir, outdir)
    if ((limit is not None and limit < 0) or concurrency < 1 or max_input_chars < 0
            or max_output_tokens < 1 or not 0 <= validation_fraction <= 1):
        raise DataError("无效的构建限额或验证集比例")
    with synthesis_lock(workdir):
        stages = {"process": process(inputs, workdir), "repair": repair(workdir)}
        if not offline and limit != 0:
            if complete is not None:
                stages["synthesize"] = await _synthesize(workdir, complete, model, limit=limit, concurrency=concurrency,
                                                        max_input_chars=max_input_chars, repair_only=True)
            else:
                stages["synthesize"] = await configured_synthesis(argparse.Namespace(
                    command="build", workdir=workdir, model=model, preset=preset, limit=limit,
                    concurrency=concurrency, max_input_chars=max_input_chars, max_output_tokens=max_output_tokens))
        stages["evidence"] = import_evidence(workdir, evidence) if evidence else {"imported": 0}
        stages["validate"] = validate_workdir(workdir)
        return build_bundle(workdir, outdir, validation_fraction, stages)


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
    repairing = commands.add_parser("repair", help="默认全量模型修复并复用真实修订，可产生 API 费用")
    repairing.add_argument("--workdir", required=True)
    building = commands.add_parser("build", help="一键处理→默认全量模型修复→校验→输出蒸馏数据包")
    building.add_argument("inputs", nargs="+")
    building.add_argument("--outdir", required=True)
    building.add_argument("--workdir", help="默认使用 outdir 同级的 trajectory-work/<outdir.name>")
    building.add_argument("--validation-fraction", type=float, default=0.1)
    validating = commands.add_parser("validate", help="重验样本并接入已有真实测试报告；不执行代码")
    validating.add_argument("--workdir", required=True)
    validating.add_argument("--evidence", action="append", default=[], help="可信测试报告 JSONL，可重复指定")
    for command in (synthesis, repairing, building):
        command.add_argument("--model")
        command.add_argument("--preset")
        command.add_argument("--limit", type=int, default=None, help="默认不限数量；N 为本次上限，0 禁用模型")
        command.add_argument("--concurrency", type=int, default=2)
        command.add_argument("--max-output-tokens", type=int, default=4096)
        command.add_argument("--max-input-chars", type=int, default=0)
    for command in (repairing, building):
        mode = command.add_mutually_exclusive_group()
        mode.add_argument("--synthesize", action="store_true", default=True,
                          help="调用配置模型修复（默认启用），可产生 API 费用")
        mode.add_argument("--offline", action="store_false", dest="synthesize",
                          help="仅离线复用真实修订，不加载模型或调用 API")
        command.add_argument("--evidence", action="append", default=[], help="已有真实测试报告 JSONL")
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
        elif args.command == "validate":
            with synthesis_lock(Path(args.workdir).resolve()):
                imported = import_evidence(args.workdir, args.evidence) if args.evidence else {"imported": 0}
                output = {"evidence": imported, "validate": validate_workdir(args.workdir)}
        else:
            if ((args.limit is not None and args.limit < 0) or args.concurrency < 1
                    or args.max_output_tokens < 1 or args.max_input_chars < 0):
                raise DataError("合成限额不能为负；并发和输出 token 上限必须为正")
            if args.workdir:
                args.workdir = str(Path(args.workdir).expanduser().resolve())
            if args.command == "build":
                outdir = Path(args.outdir).expanduser().resolve()
                args.workdir = args.workdir or str(outdir.parent / "trajectory-work" / outdir.name)
                check_build_paths(args.inputs, Path(args.workdir).expanduser().resolve(), outdir)
                if not 0 <= args.validation_fraction <= 1:
                    raise DataError("validation-fraction 必须位于 [0, 1]")
            async def execute():
                if args.command == "synthesize" and args.limit == 0:
                    return {"selected": 0}
                with synthesis_lock(Path(args.workdir).expanduser().resolve()):
                    stages = {}
                    if args.command == "build":
                        stages["process"] = process(args.inputs, args.workdir)
                    if args.command in {"repair", "build"}:
                        stages["repair"] = repair(args.workdir)
                    if args.command == "synthesize" or args.synthesize:
                        stages["synthesize"] = await configured_synthesis(args)
                    if args.command == "synthesize":
                        return stages["synthesize"]
                    stages["evidence"] = (import_evidence(args.workdir, args.evidence)
                                          if args.evidence else {"imported": 0})
                    stages["validate"] = validate_workdir(args.workdir)
                    if args.command == "build":
                        return build_bundle(args.workdir, outdir, args.validation_fraction, stages)
                    return stages
            output = asyncio.run(execute())
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0
    except (DataError, OSError, ValueError) as exc:
        argument_parser.error(str(exc))
