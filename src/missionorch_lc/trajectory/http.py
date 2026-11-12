"""在 SDK 消费前旁路采集 HTTP 响应；不替换代理、超时或重试策略。"""
from __future__ import annotations

import json
import time
from functools import wraps
from typing import Any

import httpx

from .core import _CONTEXT, capture_failure, emit, get_recorder, normalize, uid
from .stream import SafeFrames


def guarded(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception:
            capture_failure()
            return None
    return wrapped


def merge_events(events: list[dict]) -> dict:
    content, reasoning, summaries = [], [], []
    tools: dict[Any, dict] = {}
    usage, finish, blocks, details = {}, None, {}, []
    completed = None
    for event in events:
        if not isinstance(event, dict):
            continue
        kind = event.get("type", "")
        if kind in {"response.completed", "response.incomplete"}:
            completed = event.get("response")
        if kind == "response.output_text.delta":
            content.append(event.get("delta", ""))
        if kind in {"response.reasoning_text.delta", "response.reasoning.delta"}:
            reasoning.append(event.get("delta", ""))
        if kind == "response.reasoning_summary_text.delta":
            summaries.append(event.get("delta", ""))
        if kind == "message_start":
            usage.update((event.get("message") or {}).get("usage") or {})
        if kind == "response.output_item.added":
            item = event.get("item") or {}
            if item.get("type") == "function_call":
                tools[ event.get(
                    "output_index",
                    0)] = {
                    "id": item.get(
                    "call_id")
                    or item.get(
                    "id"),
                    "type": "function",
                    "function": {
                    "name": item.get(
                    "name",
                    ""),
                    "arguments": item.get(
                    "arguments",
                    "")}}
        if kind == "response.function_call_arguments.delta":
            target = tools.setdefault( event.get(
                "output_index",
                0), {
                "id": event.get(
                "item_id",
                ""),
                "type": "function",
                "function": {
                "name": "",
                "arguments": ""}})
            target["function"]["arguments"] += event.get("delta", "")
        if kind == "content_block_start":
            block = dict(event.get("content_block", {}))
            blocks[event.get("index", 0)] = block
            if block.get("type") == "tool_use":
                tools[ event.get(
                    "index",
                    0)] = {
                    "id": block.get(
                    "id"),
                    "type": "function",
                    "function": {
                    "name": block.get(
                    "name",
                    ""),
                    "arguments": ""}}
        delta = event.get("delta") or {}
        if isinstance(delta, dict):
            if delta.get("type") == "text_delta":
                content.append(delta.get("text", ""))
            if delta.get("type") == "thinking_delta":
                reasoning.append(delta.get("thinking", ""))
                block = blocks.setdefault(event.get("index", 0), {"type": "thinking"})
                block["thinking"] = block.get("thinking", "") + delta.get("thinking", "")
            if delta.get("type") == "signature_delta":
                block = blocks.setdefault(event.get("index", 0), {"type": "thinking"})
                block["signature"] = block.get("signature", "") + delta.get("signature", "")
            if delta.get("type") == "input_json_delta":
                target = tools.setdefault( event.get(
                    "index",
                    0), {
                    "id": "",
                    "type": "function",
                    "function": {
                    "name": "",
                    "arguments": ""}})
                target["function"]["arguments"] += delta.get("partial_json", "")
            finish = delta.get("stop_reason") or finish
        for choice in event.get("choices") or []:
            delta = choice.get("delta") or {}
            content.append(delta.get("content") or "")
            reasoning.append(delta.get("reasoning_content") or "")
            detail = delta.get("reasoning_details")
            if detail:
                details.extend(detail if isinstance(detail, list) else [detail])
            finish = choice.get("finish_reason") or finish
            for part in delta.get("tool_calls") or []:
                index = part.get("index", 0)
                target = tools.setdefault( index, {
                    "id": "",
                    "type": "function",
                    "function": {
                    "name": "",
                    "arguments": ""}})
                if part.get("id"):
                    target["id"] = part["id"]
                function = part.get("function") or {}
                target["function"]["name"] += function.get("name") or ""
                target["function"]["arguments"] += function.get("arguments") or ""
        usage.update(event.get("usage") or {})
    if isinstance(completed, dict):
        return completed
    result = {
        "content": "".join(
        content),
        "reasoning_content": "".join(
        reasoning) or None,
        "tool_calls": list(
        tools.values( )),
        "thinking_blocks": [
        b for b in blocks.values( ) if b.get(
        "type") in {
        "thinking",
        "redacted_thinking"}]
        or None,
        "reasoning_details": details
        or None,
        "usage": usage,
        "finish_reason": finish
        or "stop"}
    if summaries:
        result["output"] = [{"type": "reasoning", "summary": [{"type": "summary_text", "text": "".join(summaries)}]}]
    return result


def decode_response(body: bytes, content_type: str) -> tuple[Any, dict]:
    text = body.decode("utf-8", errors="replace")
    if "event-stream" not in content_type:
        try:
            raw = json.loads(text)
        except ValueError:
            raw = {"content": text}
        return raw, normalize(raw)
    events = []
    for frame in text.replace("\r\n", "\n").split("\n\n"):
        data = "\n".join(line[5:].lstrip() for line in frame.splitlines() if line.startswith("data:"))
        if not data or data == "[DONE]":
            continue
        try:
            events.append(json.loads(data))
        except ValueError:
            events.append({"unparsed_data": data})
    # 对完整流按字段关联脱敏，只替换凭据片段，保留其他正文与原生结构。
    if get_recorder().enabled:
        sanitized = []
        safe = SafeFrames(sanitized.append)
        for event in events:
            safe.push(event)
        safe.drain(True)
        events = sanitized
    return {"stream_events": events}, normalize(merge_events(events))


class Capture:
    def __init__(self, request: httpx.Request):
        self.started = time.perf_counter()
        recorder = get_recorder()
        for key, value in request.headers.items():
            if key.lower() in {"authorization", "api-key", "x-api-key", "cookie"}:
                recorder.add_secret(value)
                if value.lower().startswith("bearer "):
                    recorder.add_secret(value[7:])
        self.context = dict(_CONTEXT.get())
        self.context.setdefault("trace_id", uid())
        self.context.setdefault("run_id", uid())
        self.context.setdefault("call_id", uid())
        self.context["attempt_id"] = uid()
        try:
            body = json.loads(request.content)
        except (ValueError, httpx.RequestNotRead):
            body = {"body_available": False}
        self.context.setdefault("model", body.get("model", "unknown") if isinstance(body, dict) else "unknown")
        self.body = bytearray()
        self.frame_buffer = bytearray()
        self.finished = False
        self.status = None
        self.content_type = ""
        self.frames = 0
        self.safe_frames = SafeFrames( lambda frame: emit(
            "http_delta",
            {
            "raw_event": frame,
            "capture_level": "http"},
            **self.context))
        emit("http_start", {"request": body, "method": request.method, "url": str(request.url)}, **self.context)

    @guarded
    def append(self, chunk: bytes):
        self.body.extend(chunk)
        if "event-stream" in self.content_type:
            self.frame_buffer.extend(chunk)
            while b"\n\n" in self.frame_buffer or b"\r\n\r\n" in self.frame_buffer:
                delimiter = b"\r\n\r\n" if b"\r\n\r\n" in self.frame_buffer else b"\n\n"
                frame, _, rest = self.frame_buffer.partition(delimiter)
                self.frame_buffer = bytearray(rest)
                self.frames += 1
                text = frame.decode("utf-8", errors="replace")
                data = "\n".join(line[5:].lstrip() for line in text.splitlines() if line.startswith("data:"))
                if data and data != "[DONE]":
                    try:
                        event = json.loads(data)
                    except ValueError:
                        event = {"text": data}
                    self.safe_frames.push(event)

    @guarded
    def end(self, complete: bool, error: BaseException | None = None):
        if self.finished:
            return
        self.finished = True
        self.safe_frames.drain(True)
        raw, response = decode_response(bytes(self.body), self.content_type)
        emit(
            "http_end",
            {
            "raw_response": raw,
            "response": response,
            "status_code": self.status,
            "complete": complete,
            "elapsed_seconds": time.perf_counter() - self.started,
            "error_type": type(
            error).__name__ if error else None,
            "error": str(
            error) if error else None,
            "capture_level": "http"},
            **self.context)


class SyncTap(httpx.SyncByteStream):
    def __init__(self, stream, capture):
        self.stream, self.capture = stream, capture

    def __iter__(self):
        try:
            for chunk in self.stream:
                self.capture.append(chunk)
                yield chunk
        except BaseException as exc:
            self.capture.end(False, exc)
            raise
        else:
            self.capture.end(True)

    def close(self):
        try:
            self.stream.close()
        finally:
            self.capture.end(False)


class AsyncTap(httpx.AsyncByteStream):
    def __init__(self, stream, capture):
        self.stream, self.capture = stream, capture

    async def __aiter__(self):
        try:
            async for chunk in self.stream:
                self.capture.append(chunk)
                yield chunk
        except BaseException as exc:
            self.capture.end(False, exc)
            raise
        else:
            self.capture.end(True)

    async def aclose(self):
        try:
            await self.stream.aclose()
        finally:
            self.capture.end(False)


def sdk_http_client(*, asynchronous=False, timeout=None):
    """沿用 OpenAI/Anthropic SDK 的连接池、重定向及默认超时。"""
    client_type = AsyncClient if asynchronous else Client
    return client_type(
        timeout=timeout if timeout is not None else httpx.Timeout(600.0, connect=5.0),
        limits=httpx.Limits(max_connections=1000, max_keepalive_connections=100),
        follow_redirects=True,
    )


class Client(httpx.Client):
    def send(self, request, **kwargs):
        if not get_recorder().enabled:
            return super().send(request, **kwargs)
        capture = guarded(Capture)(request)
        if capture is None:
            return super().send(request, **kwargs)
        try:
            response = super().send(request, **kwargs)
        except BaseException as exc:
            capture.end(False, exc)
            raise
        capture.status = response.status_code
        capture.content_type = response.headers.get("content-type", "")
        if response.is_stream_consumed:
            capture.append(response.content)
            capture.end(True)
        else:
            response.stream = SyncTap(response.stream, capture)
        return response


class AsyncClient(httpx.AsyncClient):
    async def send(self, request, **kwargs):
        if not get_recorder().enabled:
            return await super().send(request, **kwargs)
        capture = guarded(Capture)(request)
        if capture is None:
            return await super().send(request, **kwargs)
        try:
            response = await super().send(request, **kwargs)
        except BaseException as exc:
            capture.end(False, exc)
            raise
        capture.status = response.status_code
        capture.content_type = response.headers.get("content-type", "")
        if response.is_stream_consumed:
            capture.append(response.content)
            capture.end(True)
        else:
            response.stream = AsyncTap(response.stream, capture)
        return response
