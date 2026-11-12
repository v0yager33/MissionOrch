"""版本化、仅追加的本地轨迹；本模块只依赖标准库。"""
from __future__ import annotations

import atexit
import base64
import contextvars
import dataclasses
import hashlib
import json
import logging
import os
import queue
import re
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

SCHEMA_VERSION = "llm-trajectory/v1"
_CONTEXT = contextvars.ContextVar("llm_trajectory_context", default={})
_RECORDER = None
_SECRET_KEYS = {
    "apikey",
    "authorization",
    "accesstoken",
    "refreshtoken",
    "clientsecret",
    "password",
    "cookie",
    "setcookie",
    "secretkey",
    "awssecretaccesskey",
    "sessiontoken",
    "xapikey",
    "xauthtoken"}
_URL_KEYS = _SECRET_KEYS | {"token", "key", "sig", "signature", "xamzsignature", "xamzcredential", "xamzsecuritytoken"}
_LOG = logging.getLogger(__name__)


def uid() -> str:
    return uuid.uuid4().hex


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, bytes):
        return {"encoding": "base64", "data": base64.b64encode(value).decode("ascii")}
    if dataclasses.is_dataclass(value):
        return jsonable(dataclasses.asdict(value))
    if hasattr(value, "model_dump"):
        return jsonable(value.model_dump(mode="json"))
    if hasattr(type(value), "to_dict"):
        return jsonable(type(value).to_dict(value))
    return str(value)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(jsonable(value), ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def private_dir(path: str | Path) -> Path:
    """新建的每层目录均为私有，不改变既有上级工作区权限。"""
    path = Path(path)
    if not path.exists():
        private_dir(path.parent)
        path.mkdir(mode=0o700, exist_ok=True)
    return path


def safe_jsonable(value):
    try:
        return jsonable(value)
    except Exception:
        capture_failure()
        return {"capture_incomplete": True}


class Recorder:
    def __init__(self, project: str, directory: str | Path, enabled: bool = True):
        self.project = project
        self.directory = Path(directory).expanduser().resolve()
        self.enabled = enabled
        self.instance = f"{os.getpid()}-{uid()}"
        self.secrets: set[str] = set()
        self._lock = threading.Lock()
        self._sequence = 0
        self._queue = queue.Queue(maxsize=2048)
        self._thread = None
        self.errors = 0
        self.files: set[Path] = set()
        self.tool_parents: dict[tuple, str] = {}

    def add_secret(self, value: Any) -> None:
        if isinstance(value, str) and len(value) >= 6:
            with self._lock:
                self.secrets.add(value)

    def redact(self, value: Any) -> Any:
        value = jsonable(value)
        if isinstance(value, dict):
            return {k: "[REDACTED]" if _key(k) in _SECRET_KEYS else self.redact(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.redact(v) for v in value]
        if not isinstance(value, str):
            return value
        with self._lock:
            secrets = tuple(self.secrets)
        for secret in secrets:
            value = value.replace(secret, "[REDACTED]")
        value = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}", "[REDACTED]", value)
        value = re.sub(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]+=*", "Bearer [REDACTED]", value)

        def clean_url(match):
            try:
                parts = urlsplit(match.group(0))
                host = parts.netloc.rsplit("@", 1)[-1]
                query = [ ( k,
                    "[REDACTED]"
                    if _key( k) in _URL_KEYS else v) for k, v in parse_qsl( parts.query, keep_blank_values=True)]
                return urlunsplit((parts.scheme, host, parts.path, urlencode(query), parts.fragment))
            except ValueError:
                return "[REDACTED_URL]"
        return re.sub(r"https?://[^\s\"'<>]+", clean_url, value)

    def emit(self, event: str, data: Any = None, **context) -> str | None:
        if not self.enabled:
            return None
        try:
            ctx = {k: v for k, v in {**_CONTEXT.get(), **context}.items() if not k.startswith("lc_")}
            ctx.setdefault("trace_id", uid())
            ctx.setdefault("run_id", uid())
            for field in ("session_id", "parent_run_id", "call_id", "attempt_id"):
                ctx.setdefault(field, None)
            ctx = self.redact(ctx)
            event_id = uid()
            payload = self.redact(data)
            with self._lock:
                self._sequence += 1
                sequence = self._sequence
                if self._thread is None:
                    self._thread = threading.Thread(target=self._write_loop, daemon=True)
                    self._thread.start()
                row = {
                    "schema_version": SCHEMA_VERSION,
                    "project": self.project,
                    "event_id": event_id,
                    "event": event,
                    "timestamp": datetime.now(
                    timezone.utc).isoformat( ),
                    "sequence": sequence,
                    "writer_id": self.instance,
                    "collection_errors": self.errors,
                    **ctx,
                    "data": payload}
                self._queue.put(row)
            return event_id
        except Exception:
            self.errors += 1
            _LOG.warning("轨迹采集失败，业务继续；本次采集不完整")
            return None

    def _write_loop(self):
        while True:
            row = self._queue.get()
            try:
                row["collection_errors"] = max(row.get("collection_errors", 0), self.errors)
                day = row["timestamp"][:10]
                directory = self.directory / "raw" / day
                private_dir(directory)
                os.chmod(self.directory, 0o700)
                path = directory / f"{self.instance}.jsonl"
                data = json.dumps(row, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
                if len(data) > 1024 * 1024:
                    blobs = self.directory / "blobs"
                    private_dir(blobs)
                    blob_data = json.dumps(row["data"], ensure_ascii=False).encode("utf-8")
                    blob_hash = hashlib.sha256(blob_data).hexdigest()
                    # 进程实例隔离 blob 名称，排他创建，避免并发追加同一 hash。
                    blob = blobs / f"{self.instance}-{blob_hash}.json"
                    try:
                        fd = os.open(blob, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                    except FileExistsError:
                        pass
                    else:
                        try:
                            view = memoryview(blob_data)
                            while view:
                                view = view[os.write(fd, view):]
                            os.fsync(fd)
                        finally:
                            os.close(fd)
                    row["data"] = {"blob_ref": str(blob.relative_to(self.directory)), "sha256": blob_hash}
                    data = json.dumps(row, ensure_ascii=False).encode("utf-8") + b"\n"
                self._append(path, data)
                self.files.add(path)
            except Exception:
                self.errors += 1
                _LOG.warning("轨迹落盘失败，业务继续；本次采集不完整")
            finally:
                self._queue.task_done()

    @staticmethod
    def _append(path: Path, data: bytes):
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags, 0o600)
        try:
            view = memoryview(data)
            while view:
                written = os.write(fd, view)
                view = view[written:]
        finally:
            os.close(fd)

    def flush(self):
        if self._thread is not None:
            self._queue.join()
        for path in tuple(self.files):
            try:
                fd = os.open(path, os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            except OSError:
                self.errors += 1
                _LOG.warning("轨迹 flush 失败：%s", path)


def configure(project: str, directory: str | Path, enabled: bool = True) -> Recorder:
    global _RECORDER
    resolved = Path(directory).expanduser().resolve()
    if _RECORDER is not None and ( _RECORDER.project, _RECORDER.directory, _RECORDER.enabled) == ( project, resolved,
        enabled):
        return _RECORDER
    if _RECORDER is not None:
        _RECORDER.flush()
    _RECORDER = Recorder(project, resolved, enabled)
    return _RECORDER


def get_recorder() -> Recorder:
    if _RECORDER is None:
        from . import configure_default
        try:
            configure_default()
        except Exception:
            _LOG.warning("轨迹配置不可用，采集不完整；业务继续")
            configure("unconfigured", Path.cwd() / "trajectories", False)
            _RECORDER.errors += 1
    return _RECORDER


def emit(event: str, data: Any = None, **context):
    return get_recorder().emit(event, data, **context)


@contextmanager
def scope(**values):
    current = dict(_CONTEXT.get())
    current.update(values)
    current.setdefault("trace_id", uid())
    current.setdefault("run_id", uid())
    token = _CONTEXT.set(current)
    try:
        yield current
    finally:
        _CONTEXT.reset(token)


class Call:
    def __init__(self, provider: str, model: str, request: Any):
        self.provider, self.model, self.request = provider, model, request
        self.id = uid()
        self.started = time.perf_counter()
        self.done = False

    def complete(self, response: Any, capture_level="normalized"):
        if self.done:
            return
        try:
            normalized = normalize(response)
            recorder = get_recorder()
            if recorder.enabled:
                with recorder._lock:
                    for tool in normalized.get("tool_calls") or []:
                        identity = tool.get("call_id") or tool.get("id")
                        if identity:
                            recorder.tool_parents[(_CONTEXT.get().get("trace_id"), identity)] = self.id
            emit(
                "llm_end",
                {
                "response": normalized,
                "capture_level": capture_level,
                "elapsed_seconds": time.perf_counter(
                ) - self.started,
                "collection_errors": recorder.errors})
            self.done = True
        except Exception:
            capture_failure()

    def delta(self, **data):
        emit("llm_delta", data)


@contextmanager
def trace_call(provider: str, model: str, request: Any):
    call = Call(provider, model, request)
    parent = _CONTEXT.get().get("call_id") or _CONTEXT.get().get("parent_call_id")
    with scope(call_id=call.id, attempt_id=uid(), parent_call_id=parent, provider=provider, model=model):
        emit("llm_start", {"request": request})
        try:
            yield call
        except BaseException as exc:
            emit(
                "llm_error",
                {
                "error_type": type(
                exc).__name__,
                "error": str(
                exc),
                "status": "cancelled"
                if type(
                exc).__name__ == "CancelledError" else "error",
                "elapsed_seconds": time.perf_counter(
                ) - call.started})
            raise
        finally:
            if not call.done:
                emit("llm_incomplete", {"elapsed_seconds": time.perf_counter() - call.started})


def tool_parent(identity):
    recorder = get_recorder()
    with recorder._lock:
        return recorder.tool_parents.get((_CONTEXT.get().get("trace_id"), identity))


def capture_failure():
    recorder = get_recorder()
    recorder.errors += 1
    _LOG.warning("轨迹采集不完整；业务继续")


def capture_native(response: Any, **metadata):
    try:
        emit("native_response", {"raw_response": jsonable(response), "capture_level": "sdk", **metadata})
    except Exception:
        capture_failure()


def normalize(response: Any) -> dict:
    raw = jsonable(response)
    if not isinstance(raw, dict):
        raw = {"content": raw}
    message = raw
    choices = raw.get("choices") or []
    if choices:
        message = choices[0].get("message") or choices[0].get("delta") or {}
    extra = message.get("additional_kwargs") or {}
    metadata = raw.get("response_metadata") or {}
    content = message.get("content", raw.get("output_text"))
    reasoning = message.get(
        "reasoning_content") or message.get(
        "reasoning") or extra.get(
        "reasoning_content") or extra.get(
        "reasoning")
    def as_blocks(value):
        return value if isinstance(value, list) else [value] if value else []

    blocks = as_blocks(message.get("thinking_blocks") or extra.get("thinking_blocks"))
    summaries = as_blocks(message.get("reasoning_summary"))
    details = as_blocks(message.get("reasoning_details") or extra.get("reasoning_details"))
    if isinstance(content, list):
        blocks = blocks + [ b for b in content if isinstance( b, dict) and b.get(
            "type") in {
            "thinking",
            "redacted_thinking",
            "reasoning"}]
    output_blocks = raw.get("output") or []
    if isinstance(output_blocks, dict):
        output_blocks = []
    candidates = raw.get("candidates") or []
    if candidates and isinstance(candidates[0], dict):
        parts = (candidates[0].get("content") or {}).get("parts") or []
        content = parts
        blocks.extend( {
            "type": "thinking",
            "thinking": part[
            "text"],
            "native_part": part}
            for part in parts if isinstance( part, dict) and part.get(
            "thought")
            and part.get(
            "text"))
    for block in output_blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "reasoning":
            blocks.append(block)
            summaries.extend(as_blocks(block.get("summary")))
        if block.get("type") == "message" and content is None:
            content = block.get("content")
    if reasoning:
        availability = "exposed"
    elif summaries:
        availability = "summary_only"
    elif any(isinstance(b, dict) and (b.get("thinking") or b.get("text")) for b in blocks + details):
        availability = "exposed"
    elif blocks or details:
        availability = "opaque"
    else:
        availability = "not_exposed"
    tools = message.get(
        "tool_calls") or [
        b for b in output_blocks if isinstance( b, dict) and b.get(
        "type") == "function_call"]
    if not tools and isinstance(content, list):
        tools = [ {
            "id": b.get(
            "id"),
            "name": b.get(
            "name"),
            "arguments": b.get(
            "input",
            { })} for b in content if isinstance( b, dict) and b.get(
            "type") == "tool_use"]
    result = {
        "content": content,
        "reasoning_content": reasoning,
        "reasoning_details": message.get(
        "reasoning_details")
        or extra.get(
        "reasoning_details"),
        "thinking_blocks": blocks
        or None,
        "reasoning_summary": summaries
        or None,
        "reasoning_availability": availability,
        "tool_calls": tools,
        "usage": raw.get(
        "usage")
        or raw.get(
        "usage_metadata")
        or metadata.get(
        "token_usage")
        or { },
        "finish_reason": (
        choices[ 0].get(
        "finish_reason")
        if choices else None) or raw.get(
        "finish_reason")
        or raw.get(
        "stop_reason")
        or raw.get(
        "status")
        or metadata.get(
        "finish_reason")
        or "stop",
        "raw_response": raw}
    return result


def _shutdown():
    if _RECORDER is not None:
        _RECORDER.flush()


atexit.register(_shutdown)
