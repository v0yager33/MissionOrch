"""联动规划的离线契约和服务测试，不调用真实模型。"""
import asyncio
import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4

import pytest

from missionorch_lc.core.planning_jobs import PlanningJobs
from missionorch_lc.planning_server import PlanningServer
from missionorch_lc.schemas.planning import PlanningError, PlanRequest, content_hash, prepare_coa

MARKDOWN = """# COA方案：离线测试
## 方案概述
保留全部上下文。
## 实体清单 (Entity Manifest)
| ID | 名称 |
| -- | -- |
| U1 | 单元一 |
| TARGET | 背景目标 |
## 地理参考 (Geo Reference)
保留原点、边界和参考坐标。
## COA矩阵
矩阵前的约束不能丢失。
| 作战实体 | 阶段1: 出动 (就绪→出动) | 阶段2: 返回 (完成→返航) |
| -- | -- | -- |
| 单元一 (U1) | 1. 起飞<br>2. 侦察 TARGET | 1. 返航 |

矩阵后的约束也需要保留。
## 决策点
| 决策点ID | 所属阶段 | 触发条件 | 选项 |
| -- | -- | -- | -- |
| D1 | Phase_1 | 天气恶化 | 若天气恶化则返航 |
## 自定义补充
未知章节必须原样保留。
"""


def test_prepare_full_partial():
    full = prepare_coa(MARKDOWN)
    assert full["markdown"] == full["snapshot_markdown"] == MARKDOWN
    assert len(full["tasks"]) == 3
    partial = prepare_coa(MARKDOWN, [full["tasks"][1]["id"]])
    snapshot = partial["snapshot_markdown"]
    assert "1. 起飞" not in snapshot and "1. 返航" not in snapshot
    assert "侦察 TARGET" in snapshot
    for text in ["就绪→出动", "完成→返航", "背景目标", "参考坐标", "未知章节必须原样保留",
                 "矩阵前的约束不能丢失", "矩阵后的约束也需要保留"]:
        assert text in snapshot
    assert [p["phase_id"] for p in prepare_coa(snapshot)["coa"]["phases"]] == ["Phase_1", "Phase_2"]
    assert partial["validation"]  # 结构报告不把规模规则当作小任务门禁。


@pytest.mark.parametrize("source", ["action", "trigger"])
@pytest.mark.parametrize("separator", [" ", ""])
def test_partial_explicit_phase_dependency(source, separator):
    reference = f"等待{separator}Phase_1{separator}完成"
    if source == "action":
        markdown = MARKDOWN.replace("1. 返航 |", f"1. {reference}后返航 |")
    else:
        markdown = MARKDOWN.replace("完成→返航", f"{reference}→返航")
    full = prepare_coa(markdown)
    selected_id = full["tasks"][-1]["id"]
    partial = prepare_coa(markdown, [selected_id])
    assert "阶段 Phase_1（出动）" in partial["selection_note"]
    assert partial["selection_note"] in partial["snapshot_markdown"]
    assert "1. 起飞" not in partial["snapshot_markdown"]
    assert "2. 侦察 TARGET" not in partial["snapshot_markdown"]
    assert partial["selected_task_ids"] == [selected_id]
    assert partial["snapshot_hash"] == content_hash(partial["snapshot_markdown"])
    assert partial["content_hash"] == full["content_hash"] == content_hash(markdown)
    assert full["snapshot_markdown"] == markdown


@pytest.mark.parametrize("dependency", ["Phase_1:U1:0", "Phase_1:U1:1"])
@pytest.mark.parametrize("separator", [" ", ""])
def test_partial_same_phase_action_dependency(dependency, separator):
    original = prepare_coa(MARKDOWN)
    selected = original["tasks"][1 if dependency.endswith(":0") else 0]
    action = f"等待{separator}{dependency}{separator}完成后执行已选任务"
    markdown = MARKDOWN.replace(selected["text"], action)
    partial = prepare_coa(markdown, [selected["id"]])
    assert f"行动 {dependency}" in partial["selection_note"]
    assert "阶段 Phase_1" not in partial["selection_note"]
    assert partial["selected_task_ids"] == [selected["id"]]
    assert partial["selection_note"] in partial["snapshot_markdown"]
    assert partial["snapshot_hash"] == content_hash(partial["snapshot_markdown"])
    snapshot_tasks = prepare_coa(partial["snapshot_markdown"])["tasks"]
    assert [task["text"] for task in snapshot_tasks] == [action]


@pytest.mark.parametrize("reference", [
    "Phase_1:U1:0", "Phase_99", "SomePhase_1", "Phase_1_other", "Phase_1:U1:00",
    "等待前置侦察完成",
])
def test_partial_does_not_infer_or_warn_selected_unknown_reference(reference):
    markdown = MARKDOWN.replace("1. 起飞", f"1. 执行 {reference} 后起飞")
    selected_id = prepare_coa(markdown)["tasks"][0]["id"]
    partial = prepare_coa(markdown, [selected_id])
    assert "显式提及未选范围" not in partial["selection_note"]
    assert "未自动扩展" in partial["selection_note"]
    assert partial["selected_task_ids"] == [selected_id]
    assert len(prepare_coa(partial["snapshot_markdown"])["tasks"]) == 1


def test_partial_ignores_references_in_unselected_actions():
    markdown = MARKDOWN.replace("侦察 TARGET", "等待 Phase_2 和 Phase_2:U1:0 后侦察 TARGET")
    selected_id = prepare_coa(markdown)["tasks"][0]["id"]
    partial = prepare_coa(markdown, [selected_id])
    assert "显式提及未选范围" not in partial["selection_note"]
    assert partial["selected_task_ids"] == [selected_id]


@pytest.mark.parametrize("selection", [[], ["unknown"], ["Phase_1:U1:0", "Phase_1:U1:0"]])
def test_invalid_selection(selection):
    with pytest.raises(PlanningError, match="有效"):
        prepare_coa(MARKDOWN, selection)


def test_parse_errors_and_references():
    with pytest.raises(PlanningError):
        prepare_coa("too short")
    with pytest.raises(PlanningError, match="引用不存在"):
        prepare_coa(MARKDOWN.replace("D1 | Phase_1", "D1 | Phase_99"))
    with pytest.raises(PlanningError, match="重复"):
        prepare_coa(MARKDOWN.replace("\n\n矩阵后的", "\n| 单元一 (U1) | 1. 起飞 | - |\n\n矩阵后的"))


@pytest.mark.parametrize("extra", [
    "## 效果链\n| 效果ID | 达成单元 |\n| -- | -- |\n| E1 | MISSING |\n",
    "## 地理参考\n| ref_id | type |\n| -- | -- |\n| <BASE:B1> | BASE |\n| <BASE:B1> | BASE |\n",
])
def test_additional_reference_checks(extra):
    with pytest.raises(PlanningError):
        prepare_coa(MARKDOWN + extra)


def test_geo_placeholders_and_matrix_width():
    with pytest.raises(PlanningError, match="地理"):
        prepare_coa(MARKDOWN.replace("侦察 TARGET", "前往 <WAYPOINT:MISSING>"))
    with pytest.raises(PlanningError, match="列数"):
        prepare_coa(MARKDOWN.replace("| 1. 返航 |", "| 1. 返航 | 2. 多余行动 |"))
    with pytest.raises(PlanningError, match="重复"):
        prepare_coa(MARKDOWN.replace("| U1 | 单元一 |", "| U1 | 单元一 |\n| U1 | 重复 |"))


async def wait_terminal(jobs, job_id):
    for _ in range(200):
        job = jobs.get(job_id)
        if job["status"] not in {"queued", "running"}:
            return job
        await asyncio.sleep(0.005)
    raise AssertionError("作业未结束")


@pytest.mark.asyncio
async def test_jobs_capacity_progress_idempotency_cancel_restart(tmp_path):
    gate = asyncio.Event()
    calls = []

    async def fake(**kwargs):
        calls.append(kwargs["mission_input"])
        kwargs["extra_callbacks"][0].on_chain_start({}, {}, name="planner")
        await gate.wait()
        return {"markdown": MARKDOWN, "final_score": 91, "api_key": "NEVER_RETURN"}

    jobs = PlanningJobs(tmp_path, runner=fake, concurrency=1, queue_limit=1)
    request = PlanRequest(request_id=uuid4(), mission="测试")
    first = jobs.create(request)
    assert jobs.create(request)["job_id"] == first["job_id"]
    with pytest.raises(PlanningError) as conflict:
        jobs.create(PlanRequest(request_id=request.request_id, mission="其他"))
    assert conflict.value.status == 409
    second = jobs.create(PlanRequest(request_id=uuid4(), mission="等待"))
    with pytest.raises(PlanningError) as full:
        jobs.create(PlanRequest(request_id=uuid4(), mission="超出"))
    assert full.value.status == 429
    await asyncio.sleep(0.02)
    assert jobs.get(first["job_id"])["logs"][0]["stage"] == "planner"
    assert calls == ["测试"]
    jobs.cancel(second["job_id"])
    gate.set()
    done = await wait_terminal(jobs, first["job_id"])
    assert done["status"] == "completed" and "NEVER_RETURN" not in json.dumps(done)
    assert jobs.get(second["job_id"])["status"] == "cancelled"
    gate.clear()
    interrupted = jobs.create(PlanRequest(request_id=uuid4(), mission="中断"))
    await asyncio.sleep(0.01)
    await jobs.close()
    restarted = PlanningJobs(tmp_path, runner=fake)
    assert restarted.get(interrupted["job_id"])["status"] == "interrupted"
    assert restarted.create(request)["job_id"] == first["job_id"]
    await restarted.close()


@pytest.mark.asyncio
async def test_timeout_failure_and_running_cancel(tmp_path):
    async def slow(**kwargs):
        await asyncio.sleep(10)
    jobs = PlanningJobs(tmp_path, runner=slow, timeout_s=0.02)
    job = jobs.create(PlanRequest(request_id=uuid4(), mission="超时"))
    assert (await wait_terminal(jobs, job["job_id"]))["status"] == "timed_out"
    job = jobs.create(PlanRequest(request_id=uuid4(), mission="取消"))
    await asyncio.sleep(0.005)
    jobs.cancel(job["job_id"])
    assert (await wait_terminal(jobs, job["job_id"]))["status"] == "cancelled"
    await jobs.close()
    async def broken(**kwargs):
        raise RuntimeError("secret /private/path")
    jobs = PlanningJobs(tmp_path, runner=broken)
    job = jobs.create(PlanRequest(request_id=uuid4(), mission="失败"))
    result = await wait_terminal(jobs, job["job_id"])
    assert result["status"] == "failed" and "secret" not in json.dumps(result)
    await jobs.close()


def test_http_auth_prepare_and_create(tmp_path):
    async def fake(**kwargs):
        return {"markdown": MARKDOWN}
    with pytest.raises(ValueError):
        PlanningServer(("127.0.0.1", 0), "", tmp_path)
    server = PlanningServer(("127.0.0.1", 0), "offline-token", tmp_path, runner=fake)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    def request(path, body=None, auth=True):
        headers = {"Authorization": "Bearer offline-token"} if auth else {}
        data = json.dumps(body).encode() if body is not None else None
        with urlopen(Request(base + path, data=data, headers=headers), timeout=3) as response:
            return json.load(response)
    try:
        assert request("/health", auth=False)["ok"]
        with pytest.raises(HTTPError) as error:
            request("/v1/coa/prepare", {"markdown": MARKDOWN}, auth=False)
        assert error.value.code == 401
        assert request("/v1/coa/prepare", {"markdown": MARKDOWN})["markdown"] == MARKDOWN
        body = {"request_id": str(uuid4()), "mission": "离线"}
        first = request("/v1/plans", body)
        assert request("/v1/plans", body)["job_id"] == first["job_id"]
        assert request(f'/v1/plans/{first["job_id"]}')["job_id"] == first["job_id"]
        assert request(f'/v1/plans/{first["job_id"]}/cancel', {})["status"] in {"cancelled", "completed"}
        with pytest.raises(HTTPError) as error:
            request("/v1/coa/prepare", {"markdown": "bad"})
        assert error.value.code == 400
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
