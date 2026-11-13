"""任务规划 HTTP 服务。启动：python -m missionorch_lc.planning_server。"""
from __future__ import annotations

import argparse
import asyncio
import hmac
import json
import os
import re
import threading
from concurrent.futures import TimeoutError as FutureTimeoutError
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .core.planning_jobs import PlanningJobs
from .schemas.planning import MAX_CONTENT_BYTES, PlanningError, PlanRequest, PrepareRequest, prepare_coa


class PlanningServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], token: str, data_dir: Path, **job_options: Any):
        if not token.strip():
            raise ValueError("必须设置 MISSIONORCH_PLANNING_TOKEN")
        self.token = token
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True,
                                       name="planning-loop")
        self.thread.start()
        self.jobs = PlanningJobs(data_dir, **job_options)
        try:
            super().__init__(address, PlanningHandler)
        except Exception:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join()
            self.loop.close()
            raise

    def call(self, method, *args):
        async def invoke():
            return method(*args)
        return asyncio.run_coroutine_threadsafe(invoke(), self.loop).result(timeout=30)

    def server_close(self):
        if self.loop.is_running():
            asyncio.run_coroutine_threadsafe(self.jobs.close(), self.loop).result(timeout=30)
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=30)
            self.loop.close()
        super().server_close()


class PlanningHandler(BaseHTTPRequestHandler):
    server: PlanningServer

    def setup(self):
        super().setup()
        self.connection.settimeout(30)

    def log_message(self, format, *args):
        # 不记录请求路径、正文或 Authorization。
        return

    def _reply(self, status: int, data: dict):
        payload = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _body(self) -> dict:
        if self.headers.get("Transfer-Encoding"):
            raise PlanningError("invalid_request", "不支持分块请求")
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise PlanningError("invalid_request", "Content-Length 无效") from exc
        if not 0 < size <= MAX_CONTENT_BYTES + 256 * 1024:
            raise PlanningError("payload_too_large", "请求为空或超过内容上限", 413)
        data = json.loads(self.rfile.read(size))
        if not isinstance(data, dict):
            raise PlanningError("invalid_request", "请求必须是 JSON 对象")
        return data

    def _dispatch(self):
        if self.command == "GET" and self.path == "/health":
            return 200, {"ok": True, "service": "missionorch-planning", "version": 1}
        auth = self.headers.get("Authorization", "")
        if not hmac.compare_digest(auth.encode(), f"Bearer {self.server.token}".encode()):
            raise PlanningError("unauthorized", "规划服务鉴权失败", 401)
        if self.path == "/v1/plans" and self.command == "POST":
            return 202, self.server.call(self.server.jobs.create, PlanRequest.model_validate(self._body()))
        if self.path == "/v1/coa/prepare" and self.command == "POST":
            request = PrepareRequest.model_validate(self._body())
            return 200, prepare_coa(request.markdown, request.selected_task_ids)
        match = re.fullmatch(r"/v1/plans/([a-fA-F0-9-]{36})(/cancel)?", self.path)
        if match:
            job_id, cancel = match.groups()
            if self.command == "GET" and not cancel:
                return 200, self.server.call(self.server.jobs.get, job_id)
            if self.command == "POST" and cancel:
                self._body()
                return 200, self.server.call(self.server.jobs.cancel, job_id)
        raise PlanningError("not_found", "接口不存在", 404)

    def _handle(self):
        try:
            status, result = self._dispatch()
            self._reply(status, result)
        except PlanningError as exc:
            self._reply(exc.status, {"error": {"code": exc.code, "message": str(exc)}})
        except (ValidationError, ValueError, UnicodeDecodeError):
            self._reply(400, {"error": {"code": "invalid_request", "message": "请求格式无效"}})
        except FutureTimeoutError:
            self._reply(504, {"error": {"code": "service_timeout", "message": "服务操作超时"}})
        except Exception:
            self._reply(500, {"error": {"code": "internal_error", "message": "规划服务内部错误"}})

    do_GET = _handle
    do_POST = _handle


def main():
    parser = argparse.ArgumentParser(description="MissionOrch 独立规划服务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8502)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--queue-limit", type=int, default=4)
    parser.add_argument("--job-timeout", type=float, default=1800)
    args = parser.parse_args()
    token = os.environ.get("MISSIONORCH_PLANNING_TOKEN", "")
    if not token.strip():
        parser.error("必须设置 MISSIONORCH_PLANNING_TOKEN")
    with PlanningServer((args.host, args.port), token, args.data_dir,
                        concurrency=args.concurrency, queue_limit=args.queue_limit,
                        timeout_s=args.job_timeout) as server:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
