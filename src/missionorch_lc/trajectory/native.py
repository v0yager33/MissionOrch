"""可选 SDK 的原生边界；按需装配，不修改第三方模块。"""
from __future__ import annotations

import inspect
import time

from .core import (
    _CONTEXT,
    capture_failure,
    capture_native,
    emit,
    get_recorder,
    normalize,
    safe_jsonable,
    scope,
    uid,
)

_METHODS = {"generate_content", "generate_content_async", "stream_generate_content", "generate_content_stream"}


class NativeClient:
    def __init__(self, client):
        self.inner = client

    def __getattr__(self, name):
        value = getattr(self.inner, name)
        if name in {"models", "aio"}:
            return NativeClient(value)
        if name not in _METHODS:
            return value

        def end(response, complete, started, error=None):
            try:
                emit("sdk_end", {
                    "response": normalize(response),
                    "complete": complete,
                    "error_type": type(error).__name__ if error else None,
                    "error": str(error) if error else None,
                    "elapsed_seconds": time.perf_counter() - started,
                    "capture_level": "sdk",
                })
            except Exception:
                capture_failure()

        def finish(result, context, started):
            frames = []

            def publish(frame):
                frames.append(frame)
                capture_native(frame, streaming=True)

            def aggregate():
                try:
                    parts = [
                        part for frame in frames if isinstance(frame, dict)
                        for candidate in frame.get("candidates", [])
                        for part in (candidate.get("content") or {}).get("parts", [])
                    ]
                    return {"candidates": [{"content": {"parts": parts}}]} if parts else {"content": frames}
                except Exception:
                    capture_failure()
                    return {"content": frames, "capture_incomplete": True}

            if hasattr(result, "__aiter__"):
                async def asynchronous():
                    from .stream import SafeFrames
                    with scope(**context):
                        safe = SafeFrames(publish)
                        complete, error = False, None
                        try:
                            async for event in result:
                                safe.push(safe_jsonable(event))
                                yield event
                            complete = True
                        except BaseException as exc:
                            error = exc
                            raise
                        finally:
                            safe.drain(True)
                            end(aggregate(), complete, started, error)
                return asynchronous()
            if inspect.isgenerator(result) or hasattr(result, "__next__"):
                def synchronous():
                    from .stream import SafeFrames
                    with scope(**context):
                        safe = SafeFrames(publish)
                        complete, error = False, None
                        try:
                            for event in result:
                                safe.push(safe_jsonable(event))
                                yield event
                            complete = True
                        except BaseException as exc:
                            error = exc
                            raise
                        finally:
                            safe.drain(True)
                            end(aggregate(), complete, started, error)
                return synchronous()
            with scope(**context):
                capture_native(result)
                end(result, True, started)
            return result

        def invoke(*args, **kwargs):
            if not get_recorder().enabled:
                return value(*args, **kwargs)
            context = {**_CONTEXT.get(), "attempt_id": uid()}
            for field in ("trace_id", "run_id", "call_id"):
                context.setdefault(field, uid())
            started = time.perf_counter()
            with scope(**context):
                emit("sdk_start", {"method": name, "args": args, "request": kwargs, "capture_level": "sdk"})
                try:
                    result = value(*args, **kwargs)
                except BaseException as exc:
                    end({}, False, started, exc)
                    raise
            if inspect.isawaitable(result):
                async def await_result():
                    with scope(**context):
                        try:
                            response = await result
                        except BaseException as exc:
                            end({}, False, started, exc)
                            raise
                        return finish(response, context, started)
                return await_result()
            return finish(result, context, started)
        return invoke


def attach_native_clients(model):
    for name in ("client", "async_client", "_client", "_async_client"):
        client = getattr(model, name, None)
        if client is not None and not isinstance(client, NativeClient):
            setattr(model, name, NativeClient(client))
    return model
