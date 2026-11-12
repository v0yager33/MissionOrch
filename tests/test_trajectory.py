"""两个仓库共享契约的离线回归；不调用真实 API、不执行轨迹代码。"""
import asyncio
import hashlib
import importlib
import json
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest

PACKAGE = "nanobot" if (Path(__file__).parents[1] / "nanobot").is_dir() else "missionorch_lc"
core = importlib.import_module(PACKAGE + ".trajectory.core")
http = importlib.import_module(PACKAGE + ".trajectory.http")
processing = importlib.import_module(PACKAGE + ".trajectory.processing")
cli = importlib.import_module(PACKAGE + ".trajectory.cli")
stream = importlib.import_module(PACKAGE + ".trajectory.stream")


@pytest.fixture
def recorder(tmp_path):
    old = core._RECORDER
    value = core.configure(PACKAGE, tmp_path / "capture")
    yield value
    value.flush()
    core._RECORDER = old


def events(recorder):
    recorder.flush()
    return processing.read_events([recorder.directory])[0]


def test_redaction_concurrency_and_permissions(recorder):
    credential = "unit-test-credential-123456"
    recorder.add_secret(credential)
    def write(index):
        with core.scope(session_id=str(index)):
            core.emit(
                "test",
                {
                "api_key": credential,
                "text": credential,
                "url": "https://u:pw@test.invalid/?access_token=" + credential})
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(write, range(100)))
    rows = events(recorder)
    assert len(rows) == 100
    assert [r["sequence"] for r in rows] == list(range(1, 101))
    assert len({r["session_id"] for r in rows}) == 100
    assert credential not in json.dumps(rows)
    assert all( all( key in r for key in (
        "trace_id",
        "session_id",
        "run_id",
        "parent_run_id",
        "call_id",
        "attempt_id"))
        for r in rows)
    if hasattr(stat, "S_IMODE"):
        assert all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in recorder.files)


def test_disabled_has_no_files(tmp_path):
    recorder = core.Recorder("test", tmp_path / "off", False)
    assert recorder.emit("test", {"content": "value"}) is None
    recorder.flush()
    assert not recorder.directory.exists()


def test_writer_failure_does_not_break_result(recorder, monkeypatch):
    def fail(*args):
        raise OSError("mock failure")
    monkeypatch.setattr(recorder, "_append", fail)
    with core.trace_call("mock", "model", {}) as call:
        call.complete({"content": "business result"})
    recorder.flush()
    assert recorder.errors > 0


def test_blob_roundtrip_and_tamper(recorder):
    core.emit("large", {"content": "中" * 400000})
    rows = events(recorder)
    assert len(rows[0]["data"]["content"]) == 400000
    blob = next((recorder.directory / "blobs").glob("*.json"))
    blob.write_text("tampered", encoding="utf-8")
    assert processing.read_events([recorder.directory])[1]["invalid_or_partial_event"] == 1


@pytest.mark.parametrize("response,availability", [
    ({"choices": [{"message": {"content": "ok", "reasoning_content": "完整思考"}}]}, "exposed"),
    ( {
        "content": [
        {
        "type": "thinking",
        "thinking": "思考",
        "signature": "opaque"},
        {
        "type": "text",
        "text": "ok"}]},
        "exposed"),

    ({"output": [{"type": "reasoning", "summary": [{"type": "summary_text", "text": "摘要"}]}]}, "summary_only"),
    ({"output": [{"type": "reasoning", "encrypted_content": "opaque"}]}, "opaque"),
    ({"content": "ok", "usage": {"reasoning_tokens": 100}}, "not_exposed"),
    ({"content": "ok", "additional_kwargs": {"reasoning_content": "可见"}}, "exposed"),
])
def test_native_reasoning(response, availability):
    assert core.normalize(response)["reasoning_availability"] == availability


class Bytes(httpx.SyncByteStream):
    def __iter__(self):
        yield b'data: {"choices":[{"delta":{"content":"hel"}}]}\n\n'
        yield b'data: {"choices":[{"delta":{"content":"lo","reasoning_content":"think"},"finish_reason":"stop"}]}\n\n'


class AsyncBytes(httpx.AsyncByteStream):
    async def __aiter__(self):
        for chunk in Bytes():
            yield chunk


def test_sync_stream_order_and_partial(recorder):
    transport = httpx.MockTransport( lambda request: httpx.Response( 200, headers={
        "content-type": "text/event-stream"},
        stream=Bytes( )))
    with http.Client(transport=transport) as client:
        with client.stream("POST", "https://test.invalid", json={"model": "mock"}) as response:
            assert next(response.iter_raw()).startswith(b"data:")
    rows = events(recorder)
    end = next(r for r in rows if r["event"] == "http_end")
    assert end["data"]["complete"] is False
    assert end["data"]["elapsed_seconds"] >= 0
    assert end["data"]["response"]["content"] == "hel"
    assert any(r["event"] == "http_delta" and "raw_event" in r["data"] for r in rows)


def test_async_stream_and_cancel(recorder):
    async def run():
        transport = httpx.MockTransport( lambda request: httpx.Response( 200, headers={
            "content-type": "text/event-stream"},
            stream=AsyncBytes( )))
        async with http.AsyncClient(transport=transport) as client:
            async with client.stream("POST", "https://test.invalid", json={"model": "mock"}) as response:
                chunks = [chunk async for chunk in response.aiter_raw()]
                assert len(chunks) == 2
        with core.trace_call("mock", "cancelled", {}) as call:
            call.delta(content="partial")
            raise asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run())
    rows = events(recorder)
    assert next(r for r in rows if r["event"] == "http_end")["data"]["response"]["content"] == "hello"
    assert next(r for r in rows if r["event"] == "llm_error")["data"]["status"] == "cancelled"


@pytest.mark.parametrize("field", ["content", "input", "unparsed_data"])
def test_split_credentials_in_stream(recorder, field):
    recorder.add_secret("credential-abcdefgh")
    frames = []
    safe = stream.SafeFrames(frames.append)
    safe.push({"choices": [{"delta": {field: "credential-"}}]})
    assert not frames
    safe.push({"choices": [{"delta": {field: "abcdefgh 正常内容"}}]})
    safe.drain(True)
    text = "".join(frame["choices"][0]["delta"][field] for frame in frames)
    assert "credential-abcdefgh" not in text
    assert "正常内容" in text


def test_capture_failure_keeps_http_success(recorder, monkeypatch):
    def fail(*args):
        raise ValueError("bad decoder")
    monkeypatch.setattr(http, "decode_response", fail)
    with http.Client( transport=httpx.MockTransport( lambda request: httpx.Response( 200, json={
        "content": "ok"}))) as client:
        assert client.post("https://test.invalid", json=[1, 2]).json()["content"] == "ok"
    assert recorder.errors > 0


def broken_then_repaired(recorder):
    prompt = [{"role": "user", "content": "编写 Python 代码"}]
    broken = {"role": "assistant", "content": "```python\ndef bad(:\n```"}
    repaired = {"role": "assistant", "content": "```python\ndef good():\n    return 1\n```"}
    with core.scope(trace_id="task", session_id="session"):
        with core.trace_call("mock", "model", {"messages": prompt}) as first:
            first.complete(broken)
        core.emit("tool_end", {"name": "test", "status": "error", "result": "SyntaxError"}, tool_call_id="test-error")
        context = prompt + [broken, {"role": "user", "content": "SyntaxError: 请修复代码并验证"}]
        with core.trace_call("mock", "model", {"messages": context}) as second:
            second.complete(repaired)
        core.emit(
            "verification",
            {
            "passed": True,
            "artifact_hash": core.digest(
            processing.artifact( repaired)),
            "target_call_id": second.id,
            "source": "trusted_test",
            "validation_scope": "unit_test"})
    return first.id, second.id, prompt, broken, repaired


def test_recovery_survives_corrected_initial(recorder, tmp_path):
    first, second, prompt, broken, repaired = broken_then_repaired(recorder)
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in recorder.files} if recorder.files else {}
    work = tmp_path / "work"
    recorder.flush()
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in recorder.files}
    stats = processing.process([recorder.directory], work)
    assert stats["calls"] == 2
    originals = processing.load_rows(work / "original.jsonl")
    repairs = processing.load_rows(work / "recovery.jsonl")
    assert len(repairs) == 1
    assert repairs[0]["status"] == "verified"
    assert repairs[0]["messages"][1] == broken
    assert repairs[0]["assistant_loss_mask"] == [False, False, False, True]
    source = next(s for s in originals if s["call_id"] == first)
    corrected = processing.correction( source, {
        "assistant": repaired,
        "source_event_ids": source[
        "source_event_ids"],
        "changes": [
        {
        "description": "改正初稿"}],
        "verification_claims": [
        "通过测试"]},
        "mock")
    assert corrected["status"] == "candidate"
    assert corrected["validation"]["evidence"] == []
    assert corrected["invalidated_event_ids"]
    processing.write_rows(work / "synthetic.jsonl", [corrected])
    assert processing.load_rows(work / "recovery.jsonl") == repairs
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == hash_value for p, hash_value in before.items())
    counts = processing.export(work, tmp_path / "export", validation_fraction=0)
    assert counts["exported_recovery_train"] == 1
    completion = processing.load_rows(tmp_path / "export/recovery.train.completion.jsonl")[0]
    assert broken in completion["prompt"]
    assert completion["completion"]["content"] == repaired["content"]
    assert completion["completion"]["role"] == "assistant"
    assert "synthetic" not in json.dumps(completion)


def test_empty_verified_and_lineage_split(recorder, tmp_path):
    with core.scope(trace_id="same"):
        for index in range(4):
            with core.trace_call("mock", "model", {"messages": [{"role": "user", "content": str(index)}]}) as call:
                call.complete({"content": "ok"})
    recorder.flush()
    work, out = tmp_path / "work", tmp_path / "out"
    recorder.flush()
    processing.process([recorder.directory], work)
    processing.export(work, out)
    assert not processing.load_rows(out / "success.train.messages.jsonl")
    processing.export(work, out, ("candidate",), 0.5)
    train = processing.load_rows(out / "success.train.messages.jsonl")
    validation = processing.load_rows(out / "success.validation.messages.jsonl")
    assert len(train) == 4 or len(validation) == 4


def test_synthesis_checkpoint_and_claim_not_evidence(recorder, tmp_path):
    broken_then_repaired(recorder)
    recorder.flush()
    work = tmp_path / "work"
    recorder.flush()
    processing.process([recorder.directory], work)
    calls = []
    async def complete(messages):
        source = json.loads(messages[-1]["content"])
        calls.append(source["id"])
        return json.dumps( {
            "assistant": {
            "role": "assistant",
            "content": "```python\nvalue = 1\n```"},
            "source_event_ids": source[
            "source_event_ids"],
            "changes": [
            ],
            "verification_claims": [
            "所有测试通过"]})
    asyncio.run(cli.synthesize(work, complete, "mock", limit=10, concurrency=2))
    count = len(calls)
    asyncio.run(cli.synthesize(work, complete, "mock", limit=10, concurrency=2))
    assert len(calls) == count == 1
    assert processing.load_rows(work / "synthetic.jsonl")[0]["status"] == "candidate"


def test_tool_protocol_and_ast_only_candidate(recorder, tmp_path):
    assert "orphan_tool_result" in processing.validate_messages(
        [ {
        "role": "tool",
        "tool_call_id": "missing",
        "content": "result"}])
    assert "invalid_tool_arguments" in processing.validate_messages(
        [ {
        "role": "assistant",
        "tool_calls": [
        {
        "id": "t",
        "function": {
        "name": "x",
        "arguments": "not json"}}]}])
    with core.trace_call("mock", "m", {"messages": [{"role": "user", "content": "code"}]}) as call:
        call.complete({"content": "```python\nx = 1\n```"})
    recorder.flush()
    processing.process([recorder.directory], tmp_path / "work")
    assert processing.load_rows(tmp_path / "work/original.jsonl")[0]["status"] == "candidate"


def test_checkpoint_result_survives_crash_window(recorder, tmp_path, monkeypatch):
    broken_then_repaired(recorder)
    work = tmp_path / "work"
    recorder.flush()
    processing.process([recorder.directory], work)
    calls = []
    async def complete(messages):
        source = json.loads(messages[-1]["content"])
        calls.append(source["id"])
        assert source["related_revisions"]
        return json.dumps( {
            "assistant": {
            "role": "assistant",
            "content": "fixed"},
            "source_event_ids": source[
            "source_event_ids"]})
    write = cli.write_rows
    def fail_checkpoint(path, rows):
        if path.name == "checkpoint.jsonl":
            raise OSError("crash window")
        write(path, rows)
    monkeypatch.setattr(cli, "write_rows", fail_checkpoint)
    with pytest.raises(OSError):
        asyncio.run(cli.synthesize(work, complete, "mock"))
    monkeypatch.setattr(cli, "write_rows", write)
    asyncio.run(cli.synthesize(work, complete, "mock"))
    assert len(calls) == 1


def test_media_is_referenced_once(recorder, tmp_path):
    media = "data:image/png;base64," + "A" * 12000
    with core.scope(trace_id="media"):
        for _ in range(2):
            with core.trace_call(
                "mock",
                "m",
                {
                "messages": [
                {
                "role": "user",
                "content": [
                {
                "type": "image_url",
                "image_url": {
                "url": media}}]}]}) as call:
                call.complete({"content": "caption"})
    work, out = tmp_path / "work", tmp_path / "out"
    recorder.flush()
    processing.process([recorder.directory], work)
    processing.export(work, out, ("candidate",), 0)
    rows = processing.load_rows(out / "success.train.messages.jsonl")
    assert len(rows) == 2
    assert media not in json.dumps(rows)
    ref = rows[0]["messages"][0]["content"][0]["image_url"]["url"]
    assert processing.load_rows(out / ref["media_ref"])[0]["value"] == media
    assert len(list((out / "media").glob("*.jsonl"))) == 1


def test_stale_verified_and_static_checks_are_not_exported(recorder, tmp_path):
    broken_then_repaired(recorder)
    work, out = tmp_path / "work", tmp_path / "out"
    recorder.flush()
    processing.process([recorder.directory], work)
    rows = processing.load_rows(work / "corrected.jsonl")
    rows[0]["messages"][-1]["content"] = "changed after validation"
    processing.write_rows(work / "corrected.jsonl", rows)
    counts = processing.export(work, out)
    assert counts["stale_artifact_skipped"] == 1
    assert not processing.verified_evidence( [ {
        "event": "verification",
        "event_id": "e",
        "data": {
        "passed": True,
        "artifact_hash": "h",
        "target_call_id": "c",
        "source": "trusted_test",
        "validation_scope": "structure"}}],
        "h",
        "c")


def test_missing_error_state_is_retained_as_incomplete(recorder, tmp_path):
    with core.scope(trace_id="compacted"):
        core.emit("tool_end", {"status": "error", "result": "SyntaxError"}, tool_call_id="t")
        with core.trace_call("mock", "m", {"messages": [{"role": "user", "content": "继续任务"}]}) as call:
            call.complete({"content": "fixed"})
    work = tmp_path / "work"
    recorder.flush()
    processing.process([recorder.directory], work)
    rows = processing.load_rows(work / "recovery.jsonl")
    assert len(rows) == 1
    assert rows[0]["status"] == "incomplete"
    assert not any(rows[0]["assistant_loss_mask"])


def test_parallel_branch_errors_are_not_spliced(recorder, tmp_path):
    with core.scope(trace_id="parallel", run_id="bad-branch"):
        core.emit("tool_end", {"status": "error", "result": "SyntaxError"})
    with core.scope(trace_id="parallel", run_id="good-branch"):
        with core.trace_call("mock", "m", {"messages": [{"role": "user", "content": "另一任务"}]}) as call:
            call.complete({"content": "ok"})
    work = tmp_path / "work"
    recorder.flush()
    processing.process([recorder.directory], work)
    assert not processing.load_rows(work / "recovery.jsonl")


def test_actual_wire_input_and_dependency_invalidation(recorder, tmp_path):
    first, second, *_ = broken_then_repaired(recorder)
    work = tmp_path / "work"
    recorder.flush()
    processing.process([recorder.directory], work)
    originals = processing.load_rows(work / "original.jsonl")
    initial = next(s for s in originals if s["call_id"] == first)
    repair = next(s for s in originals if s["call_id"] == second)
    assert repair["depends_on"][0]["sample_id"] == initial["id"]
    result = processing.correction( initial, {
        "assistant": {
        "role": "assistant",
        "content": "correct initial"},
        "source_event_ids": initial[
        "source_event_ids"]},
        "mock")
    assert set(repair["source_event_ids"]).issubset(result["invalidated_event_ids"])
    assert result["reusable_downstream_event_ids"] == []
    assert len(result["messages"]) == len(initial["messages"])


def test_native_tool_protocol_and_stream_fields(recorder):
    native = {"system": "instruction", "messages": [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "lookup", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "result"}]},
    ]}
    converted = processing.input_messages(native)
    assert converted[0]["role"] == "system"
    assert not processing.validate_messages(converted + [{"role": "assistant", "content": "ok"}])
    merged = http.merge_events([
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "完整思考"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig"}},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "t", "name": "lookup"}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{}"}},
    ])
    assert merged["thinking_blocks"][0]["signature"] == "sig"
    assert merged["thinking_blocks"][0]["thinking"] == "完整思考"
    assert merged["tool_calls"][0]["function"]["arguments"] == "{}"
    assert processing.tool_call( {
        "id": "sdk-item",
        "call_id": "real-tool",
        "name": "x",
        "arguments": "{}"})[
        "id"] == "real-tool"


def test_stream_recorder_failure_keeps_business_callback(recorder):
    def fail(frame):
        raise ValueError("capture failed")
    safe = stream.SafeFrames(fail)
    safe.push({"content": "business continues"})
    safe.drain(True)
    assert recorder.errors > 0


@pytest.mark.parametrize("details,availability", [
    ({"type": "reasoning.text", "text": "可见"}, "exposed"),
    ({"type": "reasoning.encrypted", "data": "opaque"}, "opaque"),
    (["opaque"], "opaque"),
])
def test_nonstandard_reasoning_details(details, availability):
    result = core.normalize({"content": "ok", "reasoning_details": details})
    assert result["reasoning_details"] == details
    assert result["reasoning_availability"] == availability
    assert processing.message({"role": "assistant", **result})["reasoning_details"] == details


def test_responses_parallel_tools_keep_one_assistant():
    converted = processing.input_messages({"input": [
        {"role": "user", "content": "lookup"},
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "摘要"}]},
        {"type": "function_call", "call_id": "a", "name": "lookup", "arguments": "{}"},
        {"type": "function_call", "call_id": "b", "name": "lookup", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "a", "output": "one"},
        {"type": "function_call_output", "call_id": "b", "output": "two"},
    ]})
    assert len(converted[1]["tool_calls"]) == 2
    assert converted[1]["thinking_blocks"][0]["type"] == "reasoning"
    assert not processing.validate_messages(converted + [{"role": "assistant", "content": "done"}])


def test_raw_and_blob_directory_protection(recorder, tmp_path):
    with core.trace_call("mock", "m", {"messages": [{"role": "user", "content": "task"}]}) as call:
        call.complete({"content": "ok"})
    recorder.flush()
    original = {p: p.read_bytes() for p in recorder.files}
    for directory in (recorder.directory, recorder.directory / "blobs", recorder.directory / "raw/new-day"):
        with pytest.raises(processing.DataError):
            processing.process([recorder.directory], directory)
    work = tmp_path / "work"
    processing.process([recorder.directory], work)
    with pytest.raises(processing.DataError):
        processing.export(work, recorder.directory / "blobs")
    assert all(p.read_bytes() == data for p, data in original.items())


def test_native_supplement_never_replaces_business_content(recorder, tmp_path):
    with core.trace_call("gemini", "mock", {"messages": [{"role": "user", "content": "task"}]}) as call:
        core.capture_native({"candidates": [{"content": {"parts": [
            {"text": "原生思考", "thought": True}, {"text": "原生块"},
        ]}}]})
        call.complete({"content": "SDK 解析后的真实回答"})
    recorder.flush()
    work = tmp_path / "work"
    processing.process([recorder.directory], work)
    output = processing.load_rows(work / "original.jsonl")[0]["messages"][-1]
    assert output["content"] == "SDK 解析后的真实回答"
    assert output["thinking_blocks"][0]["thinking"] == "原生思考"


def test_prompt_dependency_schema_and_hash_validation(recorder, tmp_path):
    broken_then_repaired(recorder)
    recorder.flush()
    work = tmp_path / "work"
    processing.process([recorder.directory], work)
    source = processing.load_rows(work / "original.jsonl")[0]
    proposal = json.loads(next(line for line in cli.synthesis_prompt().splitlines() if line.startswith("{")))
    identity = source["source_event_ids"][0]
    proposal["source_event_ids"] = [identity]
    proposal["depends_on"][0].update(event_id=identity, artifact_hash=source["artifact_hash"])
    assert processing.correction(source, proposal, "mock")["status"] == "candidate"
    proposal["depends_on"][0]["artifact_hash"] = "stale-hash"
    result = processing.correction(source, proposal, "mock")
    assert result["status"] == "rejected"
    assert "dependency_artifact_mismatch" in result["validation"]["issues"]
    proposal["depends_on"][0].update(artifact_hash=source["artifact_hash"], invalidated=False)
    result = processing.correction(source, proposal, "mock")
    assert "stale_dependency_not_invalidated" in result["validation"]["issues"]


def test_sdk_http_defaults_are_preserved():
    from openai import DEFAULT_TIMEOUT

    with http.sdk_http_client() as client:
        assert client.timeout == DEFAULT_TIMEOUT
        assert client.follow_redirects is True


def test_build_repairs_and_keeps_verified_recovery(recorder, tmp_path):
    first, second, _, broken, _ = broken_then_repaired(recorder)
    recorder.flush()
    raw = {path: path.read_bytes() for path in recorder.files}
    work, out = tmp_path / "work", tmp_path / "bundle"
    result = asyncio.run(cli.build([recorder.directory], work, out, validation_fraction=0, offline=True))
    assert result["ready"] is True
    assert result["verified"]["dataset_train"] == 1
    assert result["candidates"]["dataset_train"] == 1
    row = processing.load_rows(out / "train.messages.jsonl")[0]
    assert row["call_id"] == second and row["kind"] == "recovery"
    assert row["messages"][1] == broken
    assert row["assistant_loss_mask"] == [False, False, False, True]
    reused = processing.load_rows(work / "repairs.jsonl")[0]
    assert reused["call_id"] == first and reused["status"] == "candidate"
    assert reused["validation"]["evidence"] == []
    assert reused["invalidated_event_ids"]
    assert result["stages"]["repair"]["recovery_segments_preserved"] == 1
    for item in result["files"]:
        assert hashlib.sha256((out / item["path"]).read_bytes()).hexdigest() == item["sha256"]
    assert all(path.read_bytes() == value for path, value in raw.items())


def test_build_without_evidence_is_honestly_empty(recorder, tmp_path, monkeypatch):
    with core.trace_call("mock", "m", {"messages": [{"role": "user", "content": "task"}]}) as call:
        call.complete({"content": "ok"})
    recorder.flush()
    async def forbidden(*args):
        raise AssertionError("离线 build 不得加载配置模型")
    monkeypatch.setattr(cli, "model_client", forbidden)
    out = tmp_path / "bundle"
    assert cli.main(["build", str(recorder.directory), "--outdir", str(out),
                     "--validation-fraction", "0", "--offline"]) == 0
    manifest = processing.load_rows(out / "manifest.jsonl")[0]
    assert manifest["ready"] is False
    assert processing.load_rows(out / "train.messages.jsonl") == []
    assert len(processing.load_rows(out / "candidates/train.messages.jsonl")) == 1
    assert (tmp_path / "trajectory-work/bundle/original.jsonl").exists()
    assert not (tmp_path / "bundle-work").exists()


def test_model_build_repairs_partial_and_resumes(recorder, tmp_path):
    with core.scope(trace_id="partial"):
        with core.trace_call("mock", "m", {"messages": [{"role": "user", "content": "code"}]}) as call:
            call.complete({"content": "```python\nx =", "finish_reason": "length"})
    recorder.flush()
    calls = []
    async def complete(messages):
        source = json.loads(messages[-1]["content"])
        calls.append(source["id"])
        return json.dumps({"assistant": {"role": "assistant", "content": "```python\nx = 1\n```"},
                           "source_event_ids": source["source_event_ids"], "verification_claims": ["已通过测试"]})
    work, out = tmp_path / "work", tmp_path / "bundle"
    args = dict(complete=complete, model="mock", validation_fraction=0)
    result = asyncio.run(cli.build([recorder.directory], work, out, **args))
    assert result["ready"] is False and result["pending_repairs"] == 0
    assert processing.load_rows(work / "synthetic.jsonl")[0]["status"] == "candidate"
    asyncio.run(cli.build([recorder.directory], work, out, **args))
    assert len(calls) == 1
    assert len(processing.load_rows(out / "candidates/train.messages.jsonl")) == 1


def test_imported_evidence_binds_current_input_and_artifact(recorder, tmp_path):
    broken_then_repaired(recorder)
    recorder.flush()
    work, out = tmp_path / "work", tmp_path / "bundle"
    asyncio.run(cli.build([recorder.directory], work, out, validation_fraction=0, offline=True))
    source = processing.load_rows(work / "repairs.jsonl")[0]
    report = {"sample_id": source["id"], "artifact_hash": source["artifact_hash"], "input_hash": source["input_hash"],
              "source": "trusted_test", "validation_scope": "unit_test", "passed": True}
    path = tmp_path / "evidence.jsonl"
    for key, value in (("artifact_hash", "wrong"), ("input_hash", "wrong"), ("validation_scope", "syntax")):
        processing.write_rows(path, [{**report, key: value}])
        with pytest.raises(processing.DataError):
            processing.import_evidence(work, [path])
    processing.write_rows(path, [report])
    assert processing.import_evidence(work, [path])["imported"] == 1
    assert processing.import_evidence(work, [path])["imported"] == 0
    processing.validate_workdir(work)
    assert processing.load_rows(work / "repairs.jsonl")[0]["status"] == "verified"
    processing.export(work, out, validation_fraction=0)
    assert len(processing.load_rows(out / "train.messages.jsonl")) == 2
    processing.write_rows(path, [{**report, "passed": False, "report_id": "failed-run"}])
    processing.import_evidence(work, [path])
    processing.validate_workdir(work)
    assert processing.load_rows(work / "repairs.jsonl")[0]["status"] == "rejected"
    processing.write_rows(path, [{**report, "report_id": "new-passed-run"}])
    processing.import_evidence(work, [path])
    processing.validate_workdir(work)
    assert processing.load_rows(work / "repairs.jsonl")[0]["status"] == "verified"
    rows = processing.load_rows(work / "repairs.jsonl")
    rows[0]["messages"][0]["content"] = "不同任务上下文"
    processing.write_rows(work / "repairs.jsonl", rows)
    processing.validate_workdir(work)
    assert processing.load_rows(work / "repairs.jsonl")[0]["status"] == "rejected"


def test_build_does_not_reuse_stale_synthesis_from_other_input(recorder, tmp_path):
    broken_then_repaired(recorder)
    recorder.flush()
    work, out = tmp_path / "work", tmp_path / "bundle"
    async def complete(messages):
        source = json.loads(messages[-1]["content"])
        return json.dumps({"assistant": {"role": "assistant", "content": "corrected"},
                           "source_event_ids": source["source_event_ids"]})
    asyncio.run(cli.build([recorder.directory], work, out, complete=complete, model="mock"))
    other = tmp_path / "other"
    other.mkdir()
    result = asyncio.run(cli.build([other], work, out))
    assert result["verified"]["dataset_train"] == result["candidates"]["dataset_train"] == 0
    assert "stale_source_events" in processing.load_rows(work / "synthetic.jsonl")[0]["validation"]["issues"]


def test_build_and_export_protect_empty_raw_directory(tmp_path):
    raw, work = tmp_path / "raw-root", tmp_path / "work"
    raw.mkdir()
    with pytest.raises(processing.DataError):
        asyncio.run(cli.build([raw], work, raw / "export"))
    assert not work.exists()
    processing.process([raw], work)
    with pytest.raises(processing.DataError):
        processing.export(work, raw / "export")
    with pytest.raises(processing.DataError):
        processing.repair(tmp_path / "missing-work")


def test_cli_model_repair_and_validation(recorder, tmp_path, monkeypatch):
    broken_then_repaired(recorder)
    recorder.flush()
    work, out = tmp_path / "work", tmp_path / "bundle"
    selected = []
    async def client(model, preset, max_output_tokens):
        selected.append(model)
        async def complete(messages):
            source = json.loads(messages[-1]["content"])
            return json.dumps({"assistant": {"role": "assistant", "content": "fixed"},
                               "source_event_ids": source["source_event_ids"]})
        return "mock", complete, None
    monkeypatch.setattr(cli, "model_client", client)
    assert cli.main(["build", str(recorder.directory), "--workdir", str(work), "--outdir", str(out),
                     "--synthesize", "--model", "mock"]) == 0
    assert selected == ["mock"]
    assert cli.main(["repair", "--workdir", str(work), "--offline"]) == 0
    assert cli.main(["validate", "--workdir", str(work)]) == 0


def test_synthetic_recovery_supervises_only_final_repair(recorder, tmp_path):
    with core.scope(trace_id="repair"):
        request = {"messages": [{"role": "user", "content": "修复代码"},
                                {"role": "assistant", "content": "```python\ndef bad(:\n```"},
                                {"role": "user", "content": "SyntaxError: 修复"}]}
        with core.trace_call("mock", "m", request) as call:
            call.complete({"content": "```python\ndef bad(:\n```"})
    recorder.flush()
    async def complete(messages):
        source = json.loads(messages[-1]["content"])
        return json.dumps({"assistant": {"role": "assistant", "content": "```python\ndef good():\n    return 1\n```"},
                           "source_event_ids": source["source_event_ids"]})
    work, out = tmp_path / "work", tmp_path / "bundle"
    asyncio.run(cli.build([recorder.directory], work, out, complete=complete, model="mock", validation_fraction=0))
    rows = processing.load_rows(out / "candidates/train.messages.jsonl")
    assert rows[0]["kind"] == "recovery" and rows[0]["repair_segment_preserved"]
    assert rows[0]["assistant_loss_mask"] == [False, False, False, True]
    assert rows[0]["messages"][:-1] == request["messages"]


def test_semantic_error_is_repaired_and_not_supervised(recorder, tmp_path):
    prompt = [{"role": "user", "content": "返回 1"}]
    broken = {"role": "assistant", "content": "```python\ndef value():\n    return 0\n```"}
    fixed = {"role": "assistant", "content": "```python\ndef value():\n    return 1\n```"}
    with core.scope(trace_id="semantic", run_id="branch"):
        with core.trace_call("mock", "m", {"messages": prompt}) as first:
            first.complete(broken)
        core.emit("tool_end", {"status": "error", "result": "AssertionError"})
        request = {"messages": prompt + [broken, {"role": "user", "content": "AssertionError"}]}
        with core.trace_call("mock", "m", request) as call:
            call.complete(fixed)
    recorder.flush()
    work, out = tmp_path / "work", tmp_path / "bundle"
    calls = []
    async def complete(messages):
        source = json.loads(messages[-1]["content"])
        calls.append(source["call_id"])
        return json.dumps({"assistant": fixed, "source_event_ids": source["source_event_ids"]})
    asyncio.run(cli.build([recorder.directory], work, out, complete=complete, model="mock", validation_fraction=0))
    assert calls == [first.id]
    initial = next(row for row in processing.load_rows(work / "corrected.jsonl") if row["call_id"] == first.id)
    assert initial["status"] == "rejected" and not any(initial["assistant_loss_mask"])
    rows = processing.load_rows(out / "candidates/train.messages.jsonl")
    assert all(row["messages"][-1]["content"] != broken["content"] for row in rows)
    assert any(broken in row["messages"][:-1] for row in rows)


def test_real_synthesis_recording_does_not_poison_resume(recorder, tmp_path):
    broken_then_repaired(recorder)
    recorder.flush()
    work, out = tmp_path / "work", tmp_path / "bundle"
    calls = []
    async def complete(messages):
        source = json.loads(messages[-1]["content"])
        calls.append(source["id"])
        text = json.dumps({"assistant": {"role": "assistant", "content": "fixed"},
                           "source_event_ids": source["source_event_ids"]})
        with core.trace_call("mock", "m", {"messages": messages}) as call:
            call.complete({"content": text})
        recorder.flush()
        return text
    for _ in range(2):
        asyncio.run(cli.build([recorder.directory], work, out, complete=complete, model="mock"))
    assert len(calls) == 1
    synthesis_events = [row for row in events(recorder) if row.get("agent") == "trajectory_synthesis"]
    assert all(row["trace_id"] != "task" for row in synthesis_events)


def test_evidence_never_crosses_project_or_context():
    event = {"event": "verification", "event_id": "v", "project": "other", "trace_id": "other",
             "data": {"passed": True, "artifact_hash": "a", "target_call_id": "c", "source": "trusted_test",
                      "validation_scope": "unit_test", "input_hash": "old"}}
    assert not processing.verified_evidence([event], "a", "c", project="current")
    assert not processing.verified_evidence([event], "a", "c", input_hash="new")


def record_repair_batch(recorder, count):
    for index in range(count):
        with core.scope(trace_id=f"partial-{index}"):
            with core.trace_call("mock", "m", {"messages": [{"role": "user", "content": str(index)}]}) as call:
                call.complete({"content": "```python\nx =", "finish_reason": "length"})
    recorder.flush()


@pytest.mark.parametrize("entry", ["build", "repair", "python"])
def test_default_repair_covers_all_and_resumes(recorder, tmp_path, monkeypatch, entry):
    record_repair_batch(recorder, 13)
    with core.scope(trace_id="healthy"):
        with core.trace_call("mock", "m", {"messages": [{"role": "user", "content": "healthy"}]}) as call:
            call.complete({"content": "ok"})
    recorder.flush()
    work, out = tmp_path / "work", tmp_path / "bundle"
    calls, active, peak = [], 0, 0
    async def client(model, preset, max_output_tokens):
        async def complete(messages):
            nonlocal active, peak
            source = json.loads(messages[-1]["content"])
            calls.append(source["id"])
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0)
            active -= 1
            return json.dumps({"assistant": {"role": "assistant", "content": "```python\nx = 1\n```"},
                               "source_event_ids": source["source_event_ids"]})
        return "mock", complete, None
    monkeypatch.setattr(cli, "model_client", client)
    if entry == "repair":
        processing.process([recorder.directory], work)
    for _ in range(2):
        if entry == "python":
            result = asyncio.run(cli.build([recorder.directory], work, out, model="mock", validation_fraction=0))
            assert result["pending_repairs"] == 0
        else:
            args = [entry, "--workdir", str(work), "--model", "mock"]
            if entry == "build":
                args += [str(recorder.directory), "--outdir", str(out)]
            assert cli.main(args) == 0
    assert len(calls) == len(set(calls)) == 13
    assert peak == 2 and active == 0
    rows = processing.load_rows(work / "synthetic.jsonl")
    assert len(rows) == 13 and all(row["status"] == "candidate" for row in rows)
    assert all(row["assistant_loss_mask"] == [False, True] for row in rows)


@pytest.mark.parametrize("entry", ["build", "repair", "python"])
def test_explicit_repair_limit_then_default_finishes_remainder(recorder, tmp_path, monkeypatch, entry):
    record_repair_batch(recorder, 13)
    work, out = tmp_path / "work", tmp_path / "bundle"
    calls = []
    async def client(*args):
        async def complete(messages):
            source = json.loads(messages[-1]["content"])
            calls.append(source["id"])
            return json.dumps({"assistant": {"role": "assistant", "content": "fixed"},
                               "source_event_ids": source["source_event_ids"]})
        return "mock", complete, None
    monkeypatch.setattr(cli, "model_client", client)
    if entry == "repair":
        processing.process([recorder.directory], work)
    for limit, expected in ((3, 3), (None, 13), (None, 13)):
        if entry == "python":
            result = asyncio.run(cli.build([recorder.directory], work, out, model="mock", limit=limit))
            assert result["pending_repairs"] == 13 - expected
        else:
            args = [entry, "--workdir", str(work), "--model", "mock"]
            if entry == "build":
                args += [str(recorder.directory), "--outdir", str(out)]
            if limit is not None:
                args += ["--limit", str(limit)]
            assert cli.main(args) == 0
        assert len(calls) == len(set(calls)) == expected


@pytest.mark.parametrize("entry", ["build", "repair", "python"])
@pytest.mark.parametrize("disabled", ["offline", "zero"])
def test_offline_and_zero_limit_never_load_model(recorder, tmp_path, monkeypatch, entry, disabled):
    record_repair_batch(recorder, 1)
    work, out = tmp_path / "work", tmp_path / "bundle"
    async def forbidden(*args):
        raise AssertionError("离线或零限额不得加载模型")
    monkeypatch.setattr(cli, "model_client", forbidden)
    if entry == "python":
        kwargs = {"offline": True} if disabled == "offline" else {"limit": 0}
        result = asyncio.run(cli.build([recorder.directory], work, out, **kwargs))
        assert result["pending_repairs"] == 1
    else:
        if entry == "repair":
            processing.process([recorder.directory], work)
        args = [entry, "--workdir", str(work)]
        if entry == "build":
            args += [str(recorder.directory), "--outdir", str(out)]
        args += ["--offline"] if disabled == "offline" else ["--limit", "0"]
        assert cli.main(args) == 0
    assert not processing.load_rows(work / "synthetic.jsonl")


@pytest.mark.parametrize("entry", ["build", "repair"])
def test_no_repair_needed_never_loads_model(recorder, tmp_path, monkeypatch, entry):
    with core.trace_call("mock", "m", {"messages": [{"role": "user", "content": "healthy"}]}) as call:
        call.complete({"content": "ok"})
    recorder.flush()
    work, out = tmp_path / "work", tmp_path / "bundle"
    async def forbidden(*args):
        raise AssertionError("无待修复项不得加载模型")
    monkeypatch.setattr(cli, "model_client", forbidden)
    args = [entry, "--workdir", str(work)]
    if entry == "repair":
        processing.process([recorder.directory], work)
    else:
        args += [str(recorder.directory), "--outdir", str(out)]
    assert cli.main(args) == 0


@pytest.mark.parametrize("entry", ["cli", "python"])
def test_synthesize_default_is_unlimited_and_zero_still_disables(recorder, tmp_path, monkeypatch, entry):
    record_repair_batch(recorder, 13)
    work = tmp_path / "work"
    processing.process([recorder.directory], work)
    calls = []
    async def complete(messages):
        source = json.loads(messages[-1]["content"])
        calls.append(source["id"])
        return json.dumps({"assistant": {"role": "assistant", "content": "fixed"},
                           "source_event_ids": source["source_event_ids"]})
    async def client(*args):
        return "mock", complete, None
    async def forbidden(*args):
        raise AssertionError("零限额不得加载模型")
    monkeypatch.setattr(cli, "model_client", forbidden)
    if entry == "cli":
        assert cli.main(["synthesize", "--workdir", str(work), "--limit", "0"]) == 0
    else:
        assert asyncio.run(cli.synthesize(work, complete, "mock", limit=0))["selected"] == 0
    assert not calls
    monkeypatch.setattr(cli, "model_client", client)
    for _ in range(2):
        if entry == "cli":
            assert cli.main(["synthesize", "--workdir", str(work), "--model", "mock"]) == 0
        else:
            asyncio.run(cli.synthesize(work, complete, "mock"))
    assert len(calls) == len(set(calls)) == 13


def test_shared_contract_sources_match():
    root = Path(__file__).resolve().parents[2]
    source = root / "MissionOrch-LC/src/missionorch_lc/trajectory"
    target = root / "ScenClaw/nanobot/trajectory"
    if not source.is_dir() or not target.is_dir():
        pytest.skip("独立仓库环境；共有 golden 测试仍验证契约")
    for name in ("core.py", "http.py", "stream.py", "instrumentation.py", "processing.py", "cli.py"):
        assert (source / name).read_bytes() == (target / name).read_bytes()
