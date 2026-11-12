"""任务和工具边界的显式装饰器，不改变返回值或执行顺序。"""
from __future__ import annotations

import inspect
from functools import wraps

from .core import _CONTEXT, capture_failure, digest, emit, scope, tool_parent, uid


def run_scope(agent):
    def decorate(function):
        signature = inspect.signature(function)

        def context(args, kwargs):
            bound = signature.bind(*args, **kwargs).arguments
            spec = bound.get("spec")
            msg = bound.get("msg")
            session = bound.get("session_key") or getattr(spec, "session_key", None)
            if msg is not None:
                session = getattr(msg, "session_key", session)
            parent = _CONTEXT.get()
            return scope(run_id=uid(), parent_run_id=parent.get("run_id"),
                         session_id=session or parent.get("session_id"),
                         agent=parent.get("agent", agent) if agent.startswith("Provider") else agent,
                         run_kind=agent)

        if inspect.iscoroutinefunction(function):
            @wraps(function)
            async def asynchronous(*args, **kwargs):
                with context(args, kwargs):
                    emit("run_start", {"agent": agent})
                    try:
                        result = await function(*args, **kwargs)
                    except BaseException as exc:
                        emit("run_error", {"error_type": type(exc).__name__, "error": str(exc)})
                        raise
                    emit("run_end", {"agent": agent})
                    return result
            return asynchronous

        @wraps(function)
        def synchronous(*args, **kwargs):
            with context(args, kwargs):
                emit("run_start", {"agent": agent})
                try:
                    result = function(*args, **kwargs)
                except BaseException as exc:
                    emit("run_error", {"error_type": type(exc).__name__, "error": str(exc)})
                    raise
                emit("run_end", {"agent": agent})
                return result
        return synchronous
    return decorate


def stage_scope(agent):
    def decorate(function):
        @wraps(function)
        async def wrapped(state, *args, **kwargs):
            with scope(agent=agent, stage=agent, iteration=state.get("iteration", 0)):
                emit("stage_start", {"agent": agent, "iteration": state.get("iteration", 0)})
                return await function(state, *args, **kwargs)
        return wrapped
    return decorate


def context_boundary(function):
    """压缩前后快照单独关联，不把旧工具观察注入新的模型输入。"""
    signature = inspect.signature(function)
    @wraps(function)
    async def wrapped(*args, **kwargs):
        values = signature.bind(*args, **kwargs).arguments
        inputs = {k: v for k, v in values.items() if k in {"messages", "segment", "history"}}
        emit("context_boundary", {"method": function.__qualname__, "phase": "before", "inputs": inputs})
        result = await function(*args, **kwargs)
        emit("context_boundary", {"method": function.__qualname__, "phase": "after", "result": result})
        return result
    return wrapped


def tool_scope(function):
    signature = inspect.signature(function)

    @wraps(function)
    async def wrapped(*args, **kwargs):
        bound = signature.bind(*args, **kwargs).arguments
        call = bound.get("tool_call")
        # runner 已建立工具上下文时，registry 不重复计数。
        if call is None and _CONTEXT.get().get("tool_call_id"):
            return await function(*args, **kwargs)
        name = getattr(call, "name", None) or bound.get("name", "unknown")
        params = getattr(call, "arguments", None) or bound.get("params", {})
        identity = getattr(call, "id", None) or uid()
        with scope( tool_call_id=identity, tool_name=name, parent_call_id=tool_parent( identity) or _CONTEXT.get( ).get(
            "call_id")):
            emit("tool_start", {"name": name, "arguments": params})
            try:
                result = await function(*args, **kwargs)
            except BaseException as exc:
                emit("tool_end", {"name": name, "error_type": type(exc).__name__, "error": str(exc), "status": "error"})
                raise
            try:
                actual = result[0] if call is not None and isinstance(result, tuple) else result
                error = result[2] if call is not None and isinstance(result, tuple) else None
                emit(
                    "tool_end",
                    {
                    "name": name,
                    "result": actual,
                    "artifact_hash": digest(
                    actual),
                    "status": "error"
                    if error or ( isinstance( actual, str) and actual.startswith(
                    "Error")) else "success",
                    "error": str(
                    error) if error else None})
            except Exception:
                capture_failure()
            return result
    return wrapped
