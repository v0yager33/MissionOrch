"""pytest 公共 fixture / 路径注入。

使 `pytest` 可以直接发现 `src/` 下的 ``missionorch_lc`` 包，无需 `pip install -e .`。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# 按 planner.txt 模板写的完整方案：5 阶段 × 3 实体，八节齐全，能通过 check_coa
FULL_COA_TEXT = """# COA方案：X海域防空压制

## 一、方案概述
以电子战先行压制敌综合防空网络，随后夺取局部制空权并对 SAM 阵地实施打击，最后安全返航。

## 二、COA矩阵

| 作战实体 | 阶段1: 起飞集结 *(ON_START →)* *(完成编队集结)* | 阶段2: 出航突防 *(集结完成 →)* *(抵达进入点)* | 阶段3: 电磁压制 *(抵达进入点 →)* *(压制敌雷达)* | 阶段4: 打击交战 *(雷达压制成功 →)* *(摧毁SAM阵地)* | 阶段5: 脱离返航 *(打击完成 →)* *(安全返航)* |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 电子战编队 (B_EW_01) [EA-18G x 4] *(电子压制)* | 1. TAKEOFF <BASE:B-1> 2. FORM_UP <WAYPOINT:WP-MARSHAL> | 1. TRANSIT <WAYPOINT:WP-IP> 2. 保持EMCON静默 | 1. JAM 敌预警雷达 2. 发射反辐射导弹 | 1. 持续干扰 2. 掩护打击编队 | 1. EGRESS 2. RTB <BASE:B-1> |
| 打击编队 (B_ST_01) [F-18E x 8] *(对地打击)* | 1. TAKEOFF <BASE:B-1> 2. FORM_UP <WAYPOINT:WP-MARSHAL> | 1. TRANSIT <WAYPOINT:WP-IP> 2. 低空突防 | 1. 待命 2. 跟随电子战编队 | 1. STRIKE <TARGET:T-1> 2. 毁伤评估 | 1. EGRESS 2. RTB <BASE:B-1> |
| 预警指挥 (B_AEW_01) [E-2D x 1] *(预警指挥)* | 1. TAKEOFF <BASE:B-1> 2. 进入预警阵位 | 1. 空情监视 2. 引导编队 | 1. 指挥协同 2. 通报威胁 | 1. 引导打击 2. 监视敌拦截机 | 1. 指挥撤离 2. RTB <BASE:B-1> |

## 三、效果链

| 效果ID | 效果描述 | 衡量指标 | 达成实体 |
|--------|----------|----------|----------|
| E1 | 压制敌预警雷达 | 雷达失效数≥2 | B_EW_01 |
| E2 | 摧毁SAM阵地 | SAM阵地毁伤≥2 | B_ST_01 |
| E3 | 掌握空情 | 空情覆盖率≥90% | B_AEW_01 |

## 四、决策点

| 决策点ID | 所属阶段 | 触发条件 | 选项 |
|----------|----------|----------|------|
| DP1 | Phase_3 | 敌雷达未被压制 | 若压制成功则转入打击; 若压制失败则二次干扰 |
| DP2 | Phase_4 | 遭遇敌拦截机 | 若敌机少于4架则继续打击; 若多于4架则撤离 |

## 五、关键风险

| 风险类别 | 风险描述 | 缓解措施 | 触发阶段 |
|----------|----------|----------|----------|
| 作战 | SAM阵地机动转移 | 实时侦察更新目标 | Phase_4 |

## 六、实体清单（Entity Manifest）

| unit_id | platform_code | quantity | side | role | callsign | base | weapons | notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| B_EW_01 | EA-18G | 4 | BLUE | 电子压制 | GROWLER | <BASE:B-1> | 2xAGM-88 | 初始雷达OFF |
| B_ST_01 | F-18E | 8 | BLUE | 对地打击 | HORNET | <BASE:B-1> | 4xJSOW | — |
| B_AEW_01 | E-2D | 1 | BLUE | 预警指挥 | HAWKEYE | <BASE:B-1> | — | — |

## 七、地理参考（Geo Reference）

| ref_id | type | name | side | hint | notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| <BASE:B-1> | BASE | 航母 | BLUE | X海域东南300nm | 起降平台 |
| <WAYPOINT:WP-MARSHAL> | WAYPOINT | 集结点 | BLUE | 距航母50nm | 编队集结 |
| <WAYPOINT:WP-IP> | WAYPOINT | 进入点 | BLUE | 距目标区100nm | 突防入口 |
| <TARGET:T-1> | TARGET | S-400阵地 | RED | 38.5N 124.2E | 主要目标 |

## 八、预期战果（Expected Outcomes）

| 战果ID | 预期战果 | 衡量指标 | 目标值 | 关联阶段 | 关联效果 | 仿真观测量 | 来源 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| EO-1 | 摧毁或压制 S-400 SAM 阵地 | SAM阵地毁伤数 | ≥ 2 个 | Phase_4 | E2 | <TARGET:T-1> 毁伤状态 | 输入 |
| EO-2 | 蓝方战损可控 | 蓝方平台损失数 | ≤ 1 架 | 全程 | E1 | 蓝方 PLATFORM_DELETED 事件数 | 方案 |
"""


@pytest.fixture
def full_coa_text() -> str:
    return FULL_COA_TEXT
