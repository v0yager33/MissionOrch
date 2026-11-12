"""保留具体 ChatModel 的 bind/结构化输出接口，在解析器前采集。"""
from __future__ import annotations

from contextlib import contextmanager

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import Runnable, RunnableBinding

from .core import _CONTEXT, capture_failure, capture_native, emit, jsonable, safe_jsonable, scope, trace_call


class TracedChatMixin:
    @contextmanager
    def _trace(self, messages, manager, kwargs):
        if _CONTEXT.get().get("lc_model_instance") == id(self):
            yield _CONTEXT.get()["lc_active_call"]
            return
        try:
            request = {
                "messages": [
                jsonable( m) for m in messages],
                "parameters": self._get_invocation_params(
                **kwargs),
                "tags": getattr(
                manager,
                "tags",
                None),
                "metadata": getattr(
                manager,
                "metadata",
                None)}
        except Exception:
            capture_failure()
            request = {"capture_incomplete": True}
        with trace_call( type( self).__name__, getattr( self,
            "model_name",
            None) or getattr( self,
            "model",
            "unknown"),
            request) as call:
            with scope(lc_model_instance=id(self), lc_active_call=call):
                yield call

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        with self._trace(messages, run_manager, {"stop": stop, **kwargs}) as call:
            result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            for generation in result.generations:
                call.complete(generation.message)
            return result

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        with self._trace(messages, run_manager, {"stop": stop, **kwargs}) as call:
            result = await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
            for generation in result.generations:
                call.complete(generation.message)
            return result

    @staticmethod
    def _partial(message):
        from .stream import SafeFrames
        safe = SafeFrames(lambda frame: emit("llm_partial", frame))
        safe.push({"response": safe_jsonable(message)})
        safe.drain(True)

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        from .stream import SafeFrames
        with self._trace(messages, run_manager, {"stop": stop, **kwargs}) as call:
            safe = SafeFrames(lambda frame: call.delta(**frame))
            aggregate = None
            try:
                for chunk in super()._stream(messages, stop=stop, run_manager=run_manager, **kwargs):
                    safe.push({"message": safe_jsonable(chunk.message)})
                    aggregate = chunk if aggregate is None else aggregate + chunk
                    yield chunk
            finally:
                safe.drain(True)
                if aggregate is not None:
                    self._partial(aggregate.message)
            if aggregate is not None:
                call.complete(aggregate.message)

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        from .stream import SafeFrames
        with self._trace(messages, run_manager, {"stop": stop, **kwargs}) as call:
            safe = SafeFrames(lambda frame: call.delta(**frame))
            aggregate = None
            try:
                async for chunk in super()._astream(messages, stop=stop, run_manager=run_manager, **kwargs):
                    safe.push({"message": safe_jsonable(chunk.message)})
                    aggregate = chunk if aggregate is None else aggregate + chunk
                    yield chunk
            finally:
                safe.drain(True)
                if aggregate is not None:
                    self._partial(aggregate.message)
            if aggregate is not None:
                call.complete(aggregate.message)

    def _create_chat_result(self, response, *args, **kwargs):
        capture_native(response)
        return super()._create_chat_result(response, *args, **kwargs)

    def _format_output(self, response, *args, **kwargs):
        capture_native(response)
        return super()._format_output(response, *args, **kwargs)


class TrajectoryCallback(BaseCallbackHandler):
    run_inline = True
    raise_error = False

    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, inputs=None, **kwargs):
        emit(
            "tool_start",
            {
            "name": (
            serialized or { }).get(
            "name"),
            "arguments": inputs
            if inputs is not None else input_str}, tool_call_id=kwargs.get(
            "tool_call_id"),
            callback_run_id=str( run_id), parent_chain_id=str( parent_run_id) if parent_run_id else None)

    def on_tool_end(self, output, *, run_id, **kwargs):
        emit(
            "tool_end",
            {
            "result": safe_jsonable(
            output),
            "status": getattr(
            output,
            "status",
            "success")},
            tool_call_id=getattr( output,
            "tool_call_id",
            None) or kwargs.get(
            "tool_call_id"),
            callback_run_id=str( run_id))

    def on_tool_error(self, error, *, run_id, **kwargs):
        emit(
            "tool_end",
            {
            "error": str(
            error),
            "status": "error"},
            tool_call_id=kwargs.get(
            "tool_call_id"),
            callback_run_id=str( run_id))

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs):
        emit(
            "chain_start",
            {
            "name": kwargs.get(
            "name"),
            "tags": kwargs.get(
            "tags"),
            "metadata": kwargs.get(
            "metadata")},
            chain_id=str( run_id), parent_chain_id=str( parent_run_id) if parent_run_id else None)


class ScopedRunnable(RunnableBinding):
    """容错链的共同 run；不把外层包装计为模型调用。"""
    def invoke(self, input, config=None, **kwargs):
        with scope():
            return self.bound.invoke(input, self._merge_configs(config), **{**self.kwargs, **kwargs})

    async def ainvoke(self, input, config=None, **kwargs):
        with scope():
            return await self.bound.ainvoke(input, self._merge_configs(config), **{**self.kwargs, **kwargs})

    def stream(self, input, config=None, **kwargs):
        with scope():
            yield from self.bound.stream(input, self._merge_configs(config), **{**self.kwargs, **kwargs})

    async def astream(self, input, config=None, **kwargs):
        with scope():
            async for chunk in self.bound.astream(input, self._merge_configs(config), **{**self.kwargs, **kwargs}):
                yield chunk

    def batch(self, inputs, config=None, *, return_exceptions=False, **kwargs):
        return Runnable.batch(self, inputs, config, return_exceptions=return_exceptions, **kwargs)

    async def abatch(self, inputs, config=None, *, return_exceptions=False, **kwargs):
        return await Runnable.abatch(self, inputs, config, return_exceptions=return_exceptions, **kwargs)

    def transform(self, input, config=None, **kwargs):
        with scope():
            yield from self.bound.transform(input, self._merge_configs(config), **{**self.kwargs, **kwargs})

    async def atransform(self, input, config=None, **kwargs):
        with scope():
            async for chunk in self.bound.atransform(input, self._merge_configs(config), **{**self.kwargs, **kwargs}):
                yield chunk
