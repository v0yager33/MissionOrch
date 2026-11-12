"""LangChain/SDK 边界离线验证。"""
import asyncio
import json

import httpx
import pytest
from missionorch_lc.core.model_router import ModelRouter
from missionorch_lc.trajectory import http
from missionorch_lc.trajectory.core import configure, scope
from missionorch_lc.trajectory.processing import read_events


@pytest.fixture
def setup_capture(tmp_path, monkeypatch):
    recorder = configure("test", tmp_path / "capture")
    count = []
    def answer(request):
        count.append(json.loads(request.content))
        return httpx.Response( 200, json={
            "id": "c",
            "object": "chat.completion",
            "model": "mock",
            "choices": [
            {
            "index": 0,
            "message": {
            "role": "assistant",
            "content": "ok",
            "reasoning_content": "原始思考",
            "reasoning_details": [
            {
            "type": "reasoning.text",
            "text": "内容"}]},
            "finish_reason": "stop"}],
            "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "total_tokens": 3}})
    sync, asynchronous = http.Client, http.AsyncClient
    monkeypatch.setattr(http, "Client", lambda **kw: sync(transport=httpx.MockTransport(answer), **kw))
    monkeypatch.setattr(http, "AsyncClient", lambda **kw: asynchronous(transport=httpx.MockTransport(answer), **kw))
    yield recorder, count
    recorder.flush()


def test_router_sync_async_bind_and_native_reasoning(setup_capture):
    recorder, requests = setup_capture
    model = ModelRouter._create_chat_model( {
        "provider": "openai_compatible",
        "model": "mock",
        "api_key": "unit-test-key",
        "base_url": "https://test.invalid",
        "max_retries": 0})
    with scope(trace_id="task", agent="test"):
        response = model.bind(temperature=0.2).invoke("test")
        assert response.content == "ok"
        assert asyncio.run(model.ainvoke("async")).content == "ok"
        tools_model = model.bind_tools( [ {
            "type": "function",
            "function": {
            "name": "lookup",
            "description": "test",
            "parameters": {
            "type": "object",
            "properties": {
            }}}}])
        assert tools_model.invoke("tools").content == "ok"
        structured = model.with_structured_output( {
            "title": "Output",
            "type": "object",
            "properties": {
            "value": {
            "type": "string"}}},
            include_raw=True)
        assert structured is not None
    recorder.flush()
    rows = read_events([recorder.directory])[0]
    assert len([r for r in rows if r["event"] == "llm_start"]) == 3
    assert len([r for r in rows if r["event"] == "http_end"]) == 3
    assert any(r["event"] == "http_end" and r["data"]["response"]["reasoning_content"] == "原始思考" for r in rows)
    assert any( r[
        "event"] == "native_response"
        and r[
        "data"][
        "raw_response"][
        "choices"][
        0][
        "message"][
        "reasoning_details"]
        for r in rows)
    assert requests[0]["temperature"] == 0.2
    assert "tools" in requests[-1]


def test_sdk_attempts_and_fallback(tmp_path):
    from langchain_openai import ChatOpenAI
    from missionorch_lc.trajectory.langchain import TracedChatMixin
    class Model(TracedChatMixin, ChatOpenAI):
        pass
    recorder = configure("test", tmp_path / "capture")
    requests = []
    def answer(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(500, json={"error": {"message": "temporary", "type": "server_error"}})
        return httpx.Response( 200, json={
            "id": "c",
            "choices": [
            {
            "index": 0,
            "message": {
            "role": "assistant",
            "content": "ok"},
            "finish_reason": "stop"}],
            "model": "mock",
            "object": "chat.completion"})
    client = http.Client(transport=httpx.MockTransport(answer))
    model = Model(model="mock", api_key="unit-test-key", http_client=client, max_retries=1)
    with scope(trace_id="retry"):
        assert model.invoke("test").content == "ok"
    recorder.flush()
    rows = read_events([recorder.directory])[0]
    assert len([r for r in rows if r["event"] == "llm_start"]) == 1
    attempts = [r for r in rows if r["event"] == "http_start"]
    assert len(attempts) == 2
    assert attempts[0]["call_id"] == attempts[1]["call_id"]
    assert attempts[0]["attempt_id"] != attempts[1]["attempt_id"]
    client.close()


def test_independent_fallback_has_one_trace(tmp_path):
    from langchain_openai import ChatOpenAI
    from missionorch_lc.trajectory.langchain import ScopedRunnable, TracedChatMixin
    class Model(TracedChatMixin, ChatOpenAI):
        pass
    recorder = configure("test", tmp_path / "capture")
    def failed(request):
        return httpx.Response(500, json={"error": {"message": "temporary"}})
    def success(request):
        return httpx.Response( 200, json={
            "id": "c",
            "object": "chat.completion",
            "model": "fallback",
            "choices": [
            {
            "index": 0,
            "message": {
            "role": "assistant",
            "content": "ok"},
            "finish_reason": "stop"}]})
    with http.Client( transport=httpx.MockTransport( failed)) as first, http.Client( transport=httpx.MockTransport(
        success)) as second:
        primary = Model(model="primary", api_key="unit-test-key", max_retries=0, http_client=first)
        backup = Model(model="fallback", api_key="unit-test-key", max_retries=0, http_client=second)
        model = ScopedRunnable(bound=primary.with_fallbacks([backup]))
        assert model.invoke("task").content == "ok"
        assert [r.content for r in model.batch(["a", "b"])] == ["ok", "ok"]
    recorder.flush()
    starts = [r for r in read_events([recorder.directory])[0] if r["event"] == "llm_start"]
    assert len(starts) == 6
    assert len({r["trace_id"] for r in starts}) == 3
    assert starts[0]["trace_id"] == starts[1]["trace_id"]
    assert starts[0]["call_id"] != starts[1]["call_id"]


def test_langchain_stream_and_cancel_do_not_preconsume(tmp_path):
    from langchain_openai import ChatOpenAI
    from missionorch_lc.trajectory.langchain import TracedChatMixin
    class Model(TracedChatMixin, ChatOpenAI):
        pass
    class Chunks(httpx.SyncByteStream):
        def __iter__(self):
            for content, finish in (("hel", None), ("lo", "stop")):
                yield (
                    "data: " + json.dumps(
                    {
                    "id": "s",
                    "object": "chat.completion.chunk",
                    "model": "mock",
                    "choices": [
                    {
                    "index": 0,
                    "delta": {
                    "content": content,
                    "reasoning_content": "think"},
                    "finish_reason": finish}]}) + "\n\n").encode(
                    )
    recorder = configure("test", tmp_path / "capture")
    with http.Client( transport=httpx.MockTransport( lambda r: httpx.Response( 200, headers={
        "content-type": "text/event-stream"},
        stream=Chunks( )))) as client:
        model = Model(model="mock", api_key="unit-test-key", max_retries=0, http_client=client)
        assert "".join(c.content for c in model.stream("complete")) == "hello"
        iterator = model.stream("partial")
        assert next(iterator).content == "hel"
        iterator.close()
    recorder.flush()
    rows = read_events([recorder.directory])[0]
    ends = [r for r in rows if r["event"] == "http_end"]
    assert ends[0]["data"]["response"]["reasoning_content"] == "thinkthink"
    assert ends[1]["data"]["complete"] is False
    assert len([r for r in rows if r["event"] == "llm_start"]) == 2


def test_optional_native_sdk_proxy_offline(tmp_path):
    from missionorch_lc.trajectory.core import trace_call
    from missionorch_lc.trajectory.native import NativeClient
    class SDK:
        def generate_content(self, **kwargs):
            return {"candidates": [{"content": {"parts": [{"text": "thinking", "thought": True}, {"text": "ok"}]}}]}
        async def generate_content_async(self, **kwargs):
            return self.generate_content(**kwargs)
        def generate_content_stream(self, **kwargs):
            yield self.generate_content(**kwargs)
    recorder = configure("test", tmp_path / "capture")
    client = NativeClient(SDK())
    with trace_call("gemini", "mock", {"messages": [{"role": "user", "content": "test"}]}) as call:
        result = client.generate_content(contents="test")
        assert list(client.generate_content_stream(contents="test"))[0] == result
        assert asyncio.run(client.generate_content_async(contents="test")) == result
        call.complete(result)
    recorder.flush()
    rows = read_events([recorder.directory])[0]
    assert len([r for r in rows if r["event"] == "sdk_start"]) == 3
    assert len({r["attempt_id"] for r in rows if r["event"] == "sdk_start"}) == 3
    sdk_ends = [r for r in rows if r["event"] == "sdk_end"]
    assert len(sdk_ends) == 3
    assert all(r["data"]["complete"] and r["data"]["elapsed_seconds"] >= 0 for r in sdk_ends)
    assert next(r for r in rows if r["event"] == "llm_end")["data"]["response"]["reasoning_availability"] == "exposed"


def test_native_standalone_partial_and_error(tmp_path):
    from missionorch_lc.trajectory.native import NativeClient
    from missionorch_lc.trajectory.processing import load_rows, process

    class SDK:
        def generate_content_stream(self, **kwargs):
            yield {"candidates": [{"content": {"parts": [{"text": "sk-"}]}}]}
            yield {"candidates": [{"content": {"parts": [{"text": "not-consumed"}]}}]}

        async def generate_content_async(self, **kwargs):
            raise ValueError("offline failure")

    recorder = configure("test", tmp_path / "capture")
    client = NativeClient(SDK())
    iterator = client.generate_content_stream(contents="task")
    assert next(iterator)["candidates"][0]["content"]["parts"][0]["text"] == "sk-"
    iterator.close()
    with pytest.raises(ValueError, match="offline failure"):
        asyncio.run(client.generate_content_async(contents="task"))
    recorder.flush()
    rows = read_events([recorder.directory])[0]
    ends = [r for r in rows if r["event"] == "sdk_end"]
    assert len(ends) == 2
    assert not any(r["data"]["complete"] for r in ends)
    assert ends[1]["data"]["error_type"] == "ValueError"
    assert "sk-" not in json.dumps(rows)
    process([recorder.directory], tmp_path / "work")
    assert all(s["status"] == "incomplete" for s in load_rows(tmp_path / "work/original.jsonl"))


def test_partial_aggregate_redacts_incomplete_credentials(tmp_path):
    from missionorch_lc.trajectory.langchain import TracedChatMixin

    recorder = configure("test", tmp_path / "capture")
    TracedChatMixin._partial({"content": "正常文字 sk-"})
    recorder.flush()
    rows = read_events([recorder.directory])[0]
    assert "sk-" not in json.dumps(rows)
    assert "正常文字" in rows[0]["data"]["response"]["content"]


def test_callback_pairs_real_tool_id(tmp_path):
    from langchain_core.tools import tool
    from missionorch_lc.trajectory.langchain import TrajectoryCallback
    from missionorch_lc.trajectory.processing import load_rows, process
    recorder = configure("test", tmp_path / "capture")
    @tool
    def lookup(query: str) -> str:
        """离线测试工具。"""
        return "actual result: " + query
    with scope(trace_id="task"):
        result = lookup.invoke( {
            "type": "tool_call",
            "id": "real-model-tool",
            "name": "lookup",
            "args": {
            "query": "test"}},
            config={
            "callbacks": [
            TrajectoryCallback( )]})
        assert result.tool_call_id == "real-model-tool"
    recorder.flush()
    rows = read_events([recorder.directory])[0]
    assert next(r for r in rows if r["event"] == "tool_end")["tool_call_id"] == "real-model-tool"
    process([recorder.directory], tmp_path / "work")
    links = load_rows(tmp_path / "work/tool-links.jsonl")
    assert all(r["tool_call_id"] == "real-model-tool" for r in links)
