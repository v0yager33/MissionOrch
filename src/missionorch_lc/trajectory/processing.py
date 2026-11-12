"""确定性恢复、依赖检查与训练导出。绝不执行轨迹里的代码或工具。"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

from .core import SCHEMA_VERSION, digest, normalize, private_dir


class DataError(ValueError):
    pass


def files_at(inputs):
    result = set()
    for item in inputs:
        path = Path(item).expanduser().resolve()
        if path.is_file():
            result.add(path)
        elif path.is_dir():
            # 目录输入只读取不可变采集区，避免误吞处理/导出文件。
            base = path / "raw" if (path / "raw").is_dir() else path
            result.update(base.rglob("*.jsonl"))
        else:
            raise DataError(f"轨迹输入不存在：{path}")
    return sorted(result)


def source_root(path):
    path = Path(path).resolve()
    for parent in path.parents:
        if parent.name == "raw":
            return parent.parent
    return path.parent


def overlaps(left, right):
    return left.is_relative_to(right) or right.is_relative_to(left)


def read_events(inputs):
    events, statistics, seen = [], Counter(), set()
    for path in files_at(inputs):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or row.get("schema_version") != SCHEMA_VERSION:
                    statistics["unsupported_schema"] += 1
                    continue
                reference = row.get("data") or {}
                if isinstance(reference, dict) and "blob_ref" in reference:
                    candidates = [parent for parent in path.parents if (parent / "blobs").is_dir()]
                    if not candidates:
                        raise DataError("关联 blob 根目录缺失")
                    root = candidates[0]
                    blob = (root / reference["blob_ref"]).resolve()
                    if not blob.is_relative_to(root / "blobs"):
                        raise DataError("关联 blob 越界")
                    raw = blob.read_bytes()
                    if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
                        raise DataError("关联 blob hash 不匹配")
                    row["data"] = json.loads(raw)
                identity = (row.get("project"), row.get("event_id"))
                if identity[1] and identity in seen:
                    statistics["duplicate_event_id"] += 1
                    continue
                seen.add(identity)
                row["source"] = {"file": str(path), "line": number}
                events.append(row)
            except (ValueError, OSError, KeyError):
                statistics["invalid_or_partial_event"] += 1
    events.sort(key=lambda r: (r.get("timestamp", ""), r.get("writer_id", ""), r.get("sequence", 0)))
    statistics["events"] = len(events)
    return events, statistics


def write_rows(path, rows):
    path = Path(path)
    private_dir(path.parent)
    os.chmod(path.parent, 0o700)
    temporary = path.with_name(path.name + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def message(value):
    value = copy.deepcopy(value)
    role = value.get(
        "role") or {
        "human": "user",
        "ai": "assistant",
        "system": "system",
        "tool": "tool"}.get(
        value.get(
        "type"))
    result = {"role": role, "content": value.get("content")}
    thinking_fields = ("reasoning_content", "reasoning_details", "reasoning_summary", "thinking_blocks",
                       "reasoning_availability")
    for key in ("name", "tool_call_id", "tool_calls", *thinking_fields):
        if key in value:
            result[key] = value[key]
    extra = value.get("additional_kwargs") or {}
    for key in ("tool_calls", *thinking_fields):
        if key not in result and key in extra:
            result[key] = extra[key]
    if result.get("tool_calls"):
        result["tool_calls"] = [tool_call(call) for call in result["tool_calls"]]
    return result


def tool_call(call):
    if "function" in call:
        return copy.deepcopy(call)
    args = call.get("arguments", call.get("args", {}))
    return {
        "id": call.get(
        "call_id")
        or call.get(
        "id"),
        "type": "function",
        "function": {
        "name": call.get(
        "name"),
        "arguments": args
        if isinstance( args, str) else json.dumps( args, ensure_ascii=False)}}


def assistant(response):
    result = {"role": "assistant", "content": response.get("content")}
    if response.get("tool_calls"):
        result["tool_calls"] = [tool_call(call) for call in response["tool_calls"]]
    for key in (
        "reasoning_content",
        "reasoning_details",
        "thinking_blocks",
        "reasoning_summary",
        "reasoning_availability"):
        if response.get(key) is not None:
            result[key] = response[key]
    return result


def artifact(message_value):
    content = message_value.get("content")
    if isinstance(content, str):
        blocks = re.findall(r"```(?:python|py)\s*\n(.*?)```", content, re.S)
        if len(blocks) == 1:
            return blocks[0].strip()
    return message_value


def validate_messages(messages):
    issues, pending = [], set()
    for index, msg in enumerate(messages):
        if msg.get("role") not in {"system", "user", "assistant", "tool", "developer"}:
            issues.append("invalid_role")
        if pending and msg.get("role") != "tool":
            issues.append("unpaired_tool_result")
            pending.clear()
        for call in msg.get("tool_calls") or []:
            if msg.get("role") != "assistant":
                issues.append("tool_call_role")
            identity = call.get("id")
            if not identity or identity in pending:
                issues.append("invalid_tool_call_id")
            pending.add(identity)
            try:
                args = call.get("function", {}).get("arguments")
                parsed = json.loads(args) if isinstance(args, str) else args
                if not isinstance(parsed, dict):
                    issues.append("invalid_tool_arguments")
            except (ValueError, TypeError):
                issues.append("invalid_tool_arguments")
        if msg.get("role") == "tool":
            if msg.get("tool_call_id") not in pending:
                issues.append("orphan_tool_result")
            pending.discard(msg.get("tool_call_id"))
        content = msg.get("content")
        if index == len(messages) - 1 and msg.get("role") == "assistant" and isinstance(content, str):
            for code in re.findall(r"```(?:python|py)\s*\n(.*?)```", content, re.S):
                try:
                    ast.parse(code)
                except SyntaxError:
                    issues.append("python_syntax")
    # 最后一条是待执行的合法 assistant 工具请求，可作为调用样本。
    if pending and messages[-1].get("role") != "assistant":
        issues.append("unpaired_tool_result")
    return sorted(set(issues))


def input_messages(request):
    """恢复实际供应商输入协议，保留原生多模态内容块。"""
    values = request.get("messages", request.get("input", request.get("contents", [])))
    if isinstance(values, str):
        values = [{"role": "user", "content": values}]
    result = []
    for value in values or []:
        if not isinstance(value, dict):
            continue
        kind = value.get("type")
        if kind == "function_call_output":
            result.append({"role": "tool", "tool_call_id": value.get("call_id"), "content": value.get("output")})
            continue
        if kind == "function_call":
            if result and result[-1].get("role") == "assistant":
                result[-1].setdefault("tool_calls", []).append(tool_call(value))
            else:
                result.append({"role": "assistant", "content": None, "tool_calls": [tool_call(value)]})
            continue
        if kind == "reasoning":
            if not result or result[-1].get("role") != "assistant":
                result.append({"role": "assistant", "content": None})
            result[-1].setdefault("thinking_blocks", []).append(copy.deepcopy(value))
            continue
        if "parts" in value:
            value = { **value,
                "role": "assistant"
                if value.get(
                "role") == "model" else value.get(
                "role",
                "user"),
                "content": value[
                "parts"]}
        converted = message(value)
        content = converted.get("content")
        if isinstance(content, list):
            tool_results = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]
            if tool_results:
                result.extend( {
                    "role": "tool",
                    "tool_call_id": b.get(
                    "tool_use_id"),
                    "content": b.get(
                    "content")}
                    for b in tool_results)
                converted["content"] = [b for b in content if b not in tool_results]
                if not converted["content"]:
                    continue
            if converted.get("role") == "assistant" and not converted.get("tool_calls"):
                converted[
                    "tool_calls"] = [
                    tool_call( {
                    "id": b.get(
                    "id"),
                    "name": b.get(
                    "name"),
                    "arguments": b.get(
                    "input",
                    { })}) for b in content if isinstance( b, dict) and b.get(
                    "type") == "tool_use"]
        result.append(converted)
    if request.get("system"):
        result.insert(0, {"role": "system", "content": request["system"]})
    if request.get("instructions"):
        result.insert(0, {"role": "system", "content": request["instructions"]})
    return result


def same_branch(event, start):
    return ( event.get(
        "run_id") == start.get(
        "run_id")
        and ( not start.get(
        "agent")
        or event.get(
        "agent") == start.get(
        "agent"))) or (
        bool( start.get(
        "parent_call_id"))
        and event.get(
        "call_id") == start.get(
        "parent_call_id"))


def verified_evidence(events, target_hash, call_id):
    evidence = []
    for event in events:
        data = event.get("data") or {}
        if event.get("event") != "verification" or not isinstance(data, dict):
            continue
        if (data.get("passed") is True and data.get("artifact_hash") == target_hash
                and data.get("target_call_id") == call_id
                and data.get("source") in {"real_execution", "trusted_test"}
                and data.get("validation_scope") not in {None, "syntax", "structure", "model_judgment"}):
            evidence.append({"event_id": event["event_id"], **data})
    return evidence


def process(inputs, workdir):
    workdir = Path(workdir).expanduser().resolve()
    for source in files_at(inputs):
        if overlaps(workdir, source_root(source)):
            raise DataError("处理目录不能与原始采集目录重叠（包括 raw 和 blobs）")
    events, stats = read_events(inputs)
    by_call, by_trace = defaultdict(list), defaultdict(list)
    for event in events:
        key = (event.get("project"), event.get("trace_id"))
        by_trace[key].append(event)
        if event.get("call_id"):
            by_call[(event.get("project"), event["call_id"])].append(event)
    originals, corrected, recovery = [], [], []
    for (project, call_id), rows in by_call.items():
        starts = [r for r in rows if r["event"] in {"llm_start", "http_start", "sdk_start"}]
        if not starts:
            continue
        start = next((r for r in starts if r["event"] == "llm_start"), starts[0])
        request = (start.get("data") or {}).get("request") or {}
        if not isinstance(request, dict):
            request = {}
        llm_ends = [r for r in rows if r["event"] == "llm_end"]
        wire = [r for r in rows if r["event"] == "http_end"]
        sdk_ends = [r for r in rows if r["event"] == "sdk_end"]
        ends = llm_ends or wire or sdk_ends
        end = ends[-1] if ends else None
        response = (end.get("data") or {}).get("response") if end else None
        # 原生 HTTP 保留第三方字段；只选择实际最终成功尝试。
        successful_wire = [ r for r in wire if ( r.get(
            "data")
            or { }).get(
            "complete")
            and 200 <= ( ( r.get(
            "data")
            or { }).get(
            "status_code")
            or 0) < 300]
        native = successful_wire[-1]["data"].get("response") if successful_wire else None
        if native is None:
            sdk_responses = [
                normalize(r["data"]["raw_response"]) for r in rows
                if r["event"] == "native_response" and r.get("data", {}).get("raw_response") is not None
                and not r["data"].get("streaming")
            ]
            native = sdk_responses[-1] if sdk_responses else None
        if native and (not response or response.get("finish_reason") != "error"):
            if response:
                # 业务实际正文/工具协议不被 wire/native 的另一种表示覆盖。
                response = dict(response)
                for key in ("content", "tool_calls", "usage", "reasoning_content", "reasoning_details",
                            "thinking_blocks", "reasoning_summary"):
                    if not response.get(key) and native.get(key):
                        response[key] = native[key]
                if native.get("reasoning_availability") not in {None, "not_exposed"}:
                    response["reasoning_availability"] = native["reasoning_availability"]
                response["raw_response"] = native.get("raw_response")
            else:
                response = native
        response = response or normalize({})
        actual_requests = [ ( r.get(
            "data")
            or { }).get(
            "request")
            for r in rows if r[
            "event"] in {
            "http_start",
            "sdk_start"}]
        actual = next( ( r for r in reversed( actual_requests) if isinstance( r, dict) and any( k in r for k in (
            "messages",
            "input",
            "contents"))),
            request)
        snapshot = input_messages(actual)
        output = assistant(response)
        sample_messages = snapshot + [output]
        issues = validate_messages(sample_messages)
        truncated = response.get("finish_reason") in {"length", "max_tokens", "incomplete"}
        failure = bool(end and end.get("data", {}).get("error_type")) or response.get(
            "finish_reason") in {
            "error",
            "content_filter",
            "refusal"} or any(
            r[
            "event"] in {
            "llm_error",
            "provider_error"}
            for r in rows)
        if end and end["event"] == "http_end":
            failure = failure or (end["data"].get("status_code") or 0) >= 400
        incomplete = not end or truncated or ( end[
            "event"] in {"http_end", "sdk_end"}
            and not end[
            "data"].get(
            "complete")) or any(
            r.get(
            "collection_errors",
            0) for r in rows)
        target_hash = digest(artifact(output))
        trace_events = by_trace[(project, start.get("trace_id"))]
        evidence = verified_evidence(trace_events, target_hash, call_id)
        if incomplete or not snapshot:
            status = "incomplete"
        elif failure or issues:
            status = "rejected"
        else:
            status = "verified" if evidence else "candidate"
        lineage = f"{project}:{start.get('session_id') or start.get('trace_id') or call_id}"
        sample = {
            "schema_version": SCHEMA_VERSION,
            "id": f"{project}:{call_id}",
            "call_id": call_id,
            "project": project,
            "trace_id": start.get(
            "trace_id"),
            "run_id": start.get(
            "run_id"),
            "parent_run_id": start.get(
            "parent_run_id"),
            "lineage": lineage,
            "kind": "original",
            "status": status,

                  "messages": sample_messages,
                      "assistant_loss_mask": [
                      False] * len( snapshot) + [ status in {
                      "candidate",
                      "verified"}],

                  "tools": actual.get(
                      "tools")
                      or request.get(
                      "tools")
                      or request.get(
                      "parameters",
                      { }).get(
                      "tools")
                      or [ ],
                      "artifact_hash": target_hash,
                      "actual_requests": actual_requests,

                  "source_event_ids": [
                      r.get(
                      "event_id")
                      for r in rows],
                      "validation": {
                      "issues": issues,
                      "scope": "protocol_and_python_syntax",
                      "evidence": evidence},
                      "events": rows,

                  "attempts": [
                      r.get(
                      "attempt_id")
                      for r in rows if r[
                      "event"] in {
                      "http_start",
                      "sdk_start"}
                      and r.get(
                      "attempt_id")]}
        sample.update( agent=start.get(
            "agent"),
            stage=start.get(
            "stage"),
            iteration=start.get(
            "iteration"),
            parent_call_id=start.get(
            "parent_call_id"))
        originals.append(sample)
        stats.update(issues)
        stats[f"status_{status}"] += 1
        if truncated:
            stats["truncated"] += 1
        if failure:
            stats["api_error_or_refusal"] += 1
        if status in {"candidate", "verified"}:
            corrected.append( { **{ k: v for k, v in sample.items( )
                if k != "events"},
                "id": sample[
                "id"] + ":corrected",
                "kind": "corrected",
                "synthetic": False,
                "changes": [
                ],
                "depends_on": [
                ]})
            # 输入快照中有真实错误反馈时，仅监督最终修复，不改写错误上下文。
            errors = [ r for r in trace_events if same_branch( r, start) and r.get(
                "timestamp",
                "") <= start.get(
                "timestamp",
                "")
                and ( r[
                "event"] in {
                "llm_error",
                "provider_error"}
                or ( r[
                "event"] in {
                "tool_end",
                "tool_result"}
                and ( r.get(
                "data")
                or { }).get(
                "status") == "error"))]
            error_messages = [ i for i, m in enumerate( snapshot) if m.get(
                "role") in {
                "user",
                "tool"}
                and isinstance( m.get(
                "content"),
                str) and re.search(
                r"(?i)error|traceback|失败|错误|syntaxerror",
                m[
                "content"])]
            if errors or error_messages:
                visible_error = bool(error_messages)
                repair = { **{ k: v for k, v in sample.items( )
                    if k != "events"},
                    "id": sample[
                    "id"] + ":recovery",
                    "kind": "recovery",
                    "error_event_ids": [
                    e.get(
                    "event_id")
                    for e in errors],
                    "error_context_indices": error_messages,
                    "repair_segment_preserved": True,
                    "repair_events": [
                    r for r in trace_events if same_branch( r, start)],
                    "error_state_visible": visible_error}
                if not visible_error:
                    # 压缩后错误状态不可见时仍保留恢复关联，但不能伪造训练 prompt。
                    repair["status"] = "incomplete"
                    repair["assistant_loss_mask"] = [False] * len(sample_messages)
                    repair["validation"] = {**sample["validation"], "issues": ["error_state_not_visible"]}
                recovery.append(repair)
    # 用输入快照与显式工具 ID 建立依赖，绝不按重复正文删除真实调用。
    for sample in originals:
        history_hashes = {digest(artifact(m)) for m in sample["messages"][:-1] if m.get("role") == "assistant"}
        parents = [ other for other in originals if other[
            "project"] == sample[
            "project"]
            and other[
            "trace_id"] == sample[
            "trace_id"]
            and other[
            "id"] != sample[
            "id"]
            and other[
            "events"][
            0][
            "timestamp"] < sample[
            "events"][
            0][
            "timestamp"]
            and other[
            "artifact_hash"] in history_hashes]
        sample[
            "depends_on"] = [
            {
            "sample_id": p[
            "id"],
            "artifact_hash": p[
            "artifact_hash"],
            "source_event_ids": p[
            "source_event_ids"]}
            for p in parents]
        sample[
            "downstream_event_ids"] = [
            e[
            "event_id"]
            for e in by_trace[ ( sample[
            "project"],
            sample[
            "trace_id"])]
            if e.get(
            "timestamp",
            "") > sample[
            "events"][
            -1].get(
            "timestamp",
            "")
            and ( e.get(
            "run_id") == sample[
            "run_id"]
            or e.get(
            "parent_call_id") == sample[
            "call_id"])]
        sample["input_hash"] = digest(sample["messages"][:-1])
        sample[
            "context_boundaries"] = [
            r[
            "event_id"]
            for r in by_trace[ ( sample[
            "project"],
            sample[
            "trace_id"])]
            if r[
            "event"] == "context_boundary"
            and same_branch( r, sample)]
    for view in corrected + recovery:
        source = next(s for s in originals if s["call_id"] == view["call_id"] and s["project"] == view["project"])
        for key in ("depends_on", "downstream_event_ids", "input_hash", "context_boundaries"):
            view[key] = source[key]
    write_rows(workdir / "original.jsonl", originals)
    write_rows(workdir / "corrected.jsonl", corrected)
    write_rows(workdir / "recovery.jsonl", recovery)
    # 事件关联视图保留没有 call_id 的工具/压缩/ReAct/子 Agent 等边界。
    write_rows(workdir / "events.jsonl", events)
    # 回调 run ID 只作执行关联，不能冒充供应商 tool_call_id。
    callback_tools = { e.get(
        "callback_run_id"): e.get(
        "tool_call_id")
        for e in events if e.get(
        "callback_run_id")
        and e.get(
        "tool_call_id")}
    model_tools = { ( s[
        "project"],
        t.get(
        "id")): s[
        "call_id"]
        for s in originals for t in s[
        "messages"][
        -1].get(
        "tool_calls",
        [ ])}
    links = []
    for e in events:
        if e["event"] not in {"tool_start", "tool_end", "tool_result", "tool_visible"}:
            continue
        identity = e.get("tool_call_id") or callback_tools.get(e.get("callback_run_id"))
        parent = model_tools.get((e.get("project"), identity))
        links.append( {
            "source_event_id": e[
            "event_id"],
            "tool_call_id": identity,
            "model_call_id": parent,
            "callback_run_id": e.get(
            "callback_run_id"),
            "capture_relation": "paired"
            if parent else "independent_or_unresolved"})
        if not identity:
            stats["unresolved_tool_id"] += 1
    write_rows(workdir / "tool-links.jsonl", links)
    stats.update({"calls": len(originals), "corrected": len(corrected), "recovery": len(recovery)})
    write_rows(workdir / "statistics.jsonl", [dict(stats)])
    return dict(stats)


def load_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def correction(source, proposal, model):
    """合成只替换目标回答。改变产物后不拼接旧的修复/观察/验证。"""
    result = copy.deepcopy({k: v for k, v in source.items() if k != "events"})
    proposed = proposal.get("assistant")
    if not isinstance(proposed, dict) or proposed.get("role") != "assistant":
        raise DataError("合成响应必须包含 assistant 对象")
    references = proposal.get("source_event_ids", [])
    if not references or not set( references).issubset( set( source[
        "source_event_ids"] + source.get(
        "related_event_ids",
        [ ]))):
        raise DataError("合成来源引用不存在或越界")
    result["messages"] = source["messages"][:-1] + [proposed]
    result["artifact_hash"] = digest(artifact(proposed))
    changed = result["artifact_hash"] != source["artifact_hash"]
    result.update( id=source[
        "id"] + ":synthetic:" + result[
        "artifact_hash"][
        :16],
        kind="corrected",
        synthetic=True, model=model,

                  changes=proposal.get(
                      "changes",
                      [ ]), depends_on=proposal.get(
                      "depends_on",
                      [ ]), source_event_ids=references,

                  invalidated_event_ids=list( dict.fromkeys( source[
                      "source_event_ids"] + source.get(
                      "downstream_event_ids",
                      [ ]))) if changed else [ ],
                      status="candidate")
    result["source_artifact_hash"] = source["artifact_hash"]
    result["dependency_change"] = "invalidate_downstream" if changed else "unchanged"
    result["reusable_downstream_event_ids"] = []
    # 后续旧观察没有独立、匹配输入与产物的执行证据时，一律不拼接进新候选。
    result["original_dependencies"] = source.get("depends_on", [])
    known = set( source[
        "source_event_ids"] + source.get(
        "downstream_event_ids",
        [ ]) + source.get(
        "related_event_ids",
        [ ]))
    expected_hashes = {event: source["artifact_hash"] for event in source["source_event_ids"]}
    for dependency in source.get("depends_on", []):
        for event in dependency.get("source_event_ids", []):
            expected_hashes[event] = dependency.get("artifact_hash")
    expected_hashes.update(source.get("related_artifact_hashes", {}))
    known.update(expected_hashes)
    dependency_issues = []
    for dependency in proposal.get("depends_on", []):
        if not isinstance(dependency, dict) or dependency.get("event_id") not in known:
            dependency_issues.append("unknown_dependency")
            continue
        expected = expected_hashes.get(dependency["event_id"])
        if not expected or dependency.get("artifact_hash") != expected:
            dependency_issues.append("dependency_artifact_mismatch")
        if changed and expected == source["artifact_hash"] and dependency.get("invalidated") is not True:
            dependency_issues.append("stale_dependency_not_invalidated")
    result.pop("related_artifact_hashes", None)
    issues = validate_messages(result["messages"]) + sorted(set(dependency_issues))
    if not source["messages"][:-1]:
        issues.append("missing_input_snapshot")
    if any(msg.get("role") == "tool" for msg in [proposed]):
        issues.append("fabricated_tool_result")
    if source["status"] == "incomplete":
        issues.append("incomplete_source")
    result["status"] = "rejected" if issues else "candidate"
    result[
        "validation"] = {
        "issues": issues,
        "scope": "protocol_and_python_syntax",
        "evidence": [
        ],
        "model_claim_is_not_evidence": True}
    result["assistant_loss_mask"] = [False] * (len(result["messages"]) - 1) + [not issues]
    return result


def externalize_media(value, outdir, assets, media=False):
    if isinstance(value, list):
        return [externalize_media(v, outdir, assets, media) for v in value]
    if isinstance(value, dict):
        media = media or value.get(
            "type") in {
            "image",
            "image_url",
            "input_image",
            "input_audio",
            "audio",
            "video",
            "media"}
        return { k: externalize_media( v, outdir, assets, media or k in {
            "image_url",
            "inline_data",
            "inlineData"})
            for k, v in value.items( )}
    if isinstance(value, str) and (value.startswith("data:") or (media and len(value) > 4096)):
        identity = hashlib.sha256(value.encode("utf-8")).hexdigest()
        reference = f"media/{identity}.jsonl"
        if identity not in assets:
            write_rows(outdir / reference, [{"value": value}])
            assets[identity] = reference
        return {"media_ref": reference, "sha256": identity, "encoding": "utf8_reference"}
    return value


def export(workdir, outdir, statuses=("verified",), validation_fraction=0.1):
    workdir, outdir = Path(workdir).resolve(), Path(outdir).resolve()
    if workdir == outdir or workdir.is_relative_to(outdir) or outdir.is_relative_to(workdir):
        raise DataError("导出目录必须独立于原始/处理产物")
    if not 0 <= validation_fraction <= 1:
        raise DataError("validation-fraction 必须位于 [0, 1]")
    source_paths = [ Path( r.get(
        "source",
        { }).get(
        "file",
        "")).resolve(
        ) for r in load_rows(
        workdir / "events.jsonl")
        if r.get(
        "source",
        { }).get(
        "file")]
    if any(overlaps(outdir, source_root(p)) for p in source_paths):
        raise DataError("导出目录不能与原始采集目录重叠（包括 raw 和 blobs）")
    samples = load_rows(
        workdir / "corrected.jsonl") + load_rows(
        workdir / "synthetic.jsonl") + load_rows(
        workdir / "recovery.jsonl")
    products, counts = defaultdict(list), Counter()
    assets = {}
    evidence_events = load_rows(workdir / "events.jsonl")
    for sample in samples:
        counts[f"available_{sample['status']}"] += 1
        if sample["status"] not in statuses:
            continue
        if validate_messages(sample["messages"]):
            counts["invalid_protocol_skipped"] += 1
            continue
        # verified 必须携带当前产物一致的可信证据，不能由可编辑状态字段单独授权。
        if sample["artifact_hash"] != digest(artifact(sample["messages"][-1])):
            counts["stale_artifact_skipped"] += 1
            continue
        if sample[
            "status"] == "verified" and not verified_evidence(
            evidence_events, sample[
            "artifact_hash"],
            sample.get(
            "call_id")):
            counts["unbacked_verified_skipped"] += 1
            continue
        split_value = int(hashlib.sha256(sample["lineage"].encode()).hexdigest()[:16], 16) / 2**64
        split = "validation" if split_value < validation_fraction else "train"
        kind = "recovery" if sample["kind"] == "recovery" else "success"
        row = { k: v for k, v in sample.items( ) if k not in {
            "events",
            "invalidated_event_ids",
            "repair_events",
            "actual_requests",
            "downstream_event_ids"}}
        row = externalize_media(row, outdir, assets)
        row["provenance_ref"] = "provenance.jsonl"
        # 显式监督掩码：所有历史 assistant（包括错误）都不监督。
        row["assistant_loss_mask"] = [False] * (len(row["messages"]) - 1) + [True]
        products[f"{kind}.{split}.messages.jsonl"].append(row)
        products[ f"{kind}.{split}.completion.jsonl"].append( { **{ k: v for k, v in row.items( ) if k not in {
            "messages",
            "assistant_loss_mask"}},
            "prompt": row[
            "messages"][
            :-1],
            "completion": row[
            "messages"][
            -1]})
        counts[f"exported_{kind}_{split}"] += 1
    for kind in ("success", "recovery"):
        for split in ("train", "validation"):
            for form in ("messages", "completion"):
                name = f"{kind}.{split}.{form}.jsonl"
                write_rows(outdir / name, products[name])
    write_rows(
        outdir / "provenance.jsonl",
        [ {
        "id": s[
        "id"],
        "source_event_ids": s[
        "source_event_ids"],
        "error_event_ids": s.get(
        "error_event_ids",
        [ ]),
        "artifact_hash": s[
        "artifact_hash"]}
        for s in samples])
    write_rows(outdir / "media-index.jsonl", [{"sha256": h, "media_ref": p} for h, p in assets.items()])
    write_rows(outdir / "statistics.jsonl", [dict(counts)])
    return dict(counts)
