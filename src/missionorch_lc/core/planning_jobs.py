"""独立规划作业：有界执行、幂等和重启恢复，不依赖 Streamlit。"""
from __future__ import annotations

import asyncio
import copy
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable
from uuid import UUID, uuid4

from langchain_core.callbacks import BaseCallbackHandler

from ..schemas.planning import PlanningError, PlanRequest, content_hash, prepare_coa

ACTIVE = {"queued", "running"}
STAGES = {"analyst", "researcher", "planner", "judge", "reflector", "finalize"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".planning-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


async def default_runner(**kwargs: Any) -> dict:
    from ..orchestrator_graph import run_graph
    return await run_graph(**kwargs)


class StageCallback(BaseCallbackHandler):
    """仅暴露阶段名，不发送模型正文、工具输入或异常中的凭据。"""

    def __init__(self, push: Callable[[str, str], None]):
        self.push = push

    def on_chain_start(self, serialized, inputs, **kwargs):
        name = kwargs.get("name") or kwargs.get("run_name") or (serialized or {}).get("name")
        if name in STAGES:
            self.push("stage_start", name)

    def on_chain_error(self, error, **kwargs):
        self.push("stage_error", "阶段执行发生错误")


class PlanningJobs:
    """所有方法在服务的一个常驻 asyncio loop 上调用。"""

    def __init__(self, data_dir: Path, *, runner: Callable[..., Awaitable[dict]] = default_runner,
                 concurrency: int = 1, queue_limit: int = 4, timeout_s: float = 1800):
        if concurrency < 1 or queue_limit < 0 or timeout_s <= 0:
            raise ValueError("作业容量或超时配置无效")
        self.directory = data_dir / "planning-jobs"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.runner, self.capacity, self.timeout_s = runner, concurrency + queue_limit, timeout_s
        self.semaphore = asyncio.Semaphore(concurrency)
        self.jobs: dict[str, dict] = {}
        self.requests: dict[str, str] = {}
        self.tasks: dict[str, asyncio.Task] = {}
        for path in self.directory.glob("*.json"):
            if path.is_symlink():
                raise ValueError("作业目录不能包含符号链接")
            job = json.loads(path.read_text(encoding="utf-8"))
            if str(UUID(job["job_id"])) != path.stem:
                raise ValueError("作业文件 ID 不一致")
            if job["status"] in ACTIVE:
                job.update(status="interrupted", updated_at=now())
                atomic_json(path, job)
            self.jobs[job["job_id"]] = job
            self.requests[job["request_id"]] = job["job_id"]

    def _persist(self, job: dict) -> None:
        job["updated_at"] = now()
        atomic_json(self.directory / f'{job["job_id"]}.json', job)

    def get(self, job_id: str) -> dict:
        try:
            job_id = str(UUID(job_id))
        except ValueError as exc:
            raise PlanningError("not_found", "作业不存在", 404) from exc
        if job_id not in self.jobs:
            raise PlanningError("not_found", "作业不存在", 404)
        return copy.deepcopy(self.jobs[job_id])

    def create(self, request: PlanRequest) -> dict:
        request_id = str(request.request_id)
        digest = content_hash(request.mission)
        if request_id in self.requests:
            job = self.jobs[self.requests[request_id]]
            if job["input_hash"] != digest:
                raise PlanningError("idempotency_conflict", "request_id 已用于不同任务", 409)
            return copy.deepcopy(job)
        if sum(job["status"] in ACTIVE for job in self.jobs.values()) >= self.capacity:
            raise PlanningError("queue_full", "规划等待队列已满", 429)
        job_id = str(uuid4())
        job = {"job_id": job_id, "request_id": request_id, "mission": request.mission,
               "input_hash": digest, "status": "queued", "created_at": now(), "logs": []}
        self._persist(job)
        self.jobs[job_id], self.requests[request_id] = job, job_id
        task = asyncio.create_task(self._execute(job))
        self.tasks[job_id] = task
        task.add_done_callback(lambda _: self.tasks.pop(job_id, None))
        return copy.deepcopy(job)

    async def _execute(self, job: dict) -> None:
        loop = asyncio.get_running_loop()

        def record(kind: str, stage: str) -> None:
            if job["status"] not in ACTIVE:
                return
            job["logs"].append({"type": kind, "stage": stage, "ts": now()})
            job["logs"] = job["logs"][-100:]
            self._persist(job)

        def push(kind: str, stage: str) -> None:
            loop.call_soon_threadsafe(record, kind, stage)

        try:
            async with self.semaphore:
                job["status"] = "running"
                self._persist(job)
                result = await asyncio.wait_for(
                    self.runner(mission_input=job["mission"], extra_callbacks=[StageCallback(push)]),
                    timeout=self.timeout_s,
                )
                markdown = result.get("markdown") or ""
                prepared = prepare_coa(markdown)
                job["result"] = {"markdown": markdown, "final_coa": prepared["coa"],
                                 "prepared": prepared, "generation": {
                                     key: result[key] for key in ("iterations", "final_score", "stop_reason")
                                     if key in result}}
                job["status"] = "completed"
        except asyncio.TimeoutError:
            job["status"] = "timed_out"
        except asyncio.CancelledError:
            if job["status"] != "interrupted":
                job["status"] = "cancelled"
        except Exception as exc:
            job["status"] = "failed"
            job["error"] = {"code": exc.code if isinstance(exc, PlanningError) else "generation_failed",
                            "message": str(exc) if isinstance(exc, PlanningError) else "规划执行失败，请查看服务日志"}
        finally:
            self._persist(job)

    def cancel(self, job_id: str) -> dict:
        job = self.jobs[self.get(job_id)["job_id"]]
        if job["status"] in ACTIVE:
            job["status"] = "cancelled"
            self._persist(job)
            if job["job_id"] in self.tasks:
                self.tasks[job["job_id"]].cancel()
        return copy.deepcopy(job)

    async def close(self) -> None:
        tasks = list(self.tasks.values())
        for job_id, task in list(self.tasks.items()):
            job = self.jobs[job_id]
            if job["status"] in ACTIVE:
                job["status"] = "interrupted"
                self._persist(job)
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
