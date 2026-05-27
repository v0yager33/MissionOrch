"""YAML 配置读写工具。

特性：
- 读取时保留原始结构与顺序（用 ruamel.yaml 优先；fallback 到 PyYAML）
- 写盘前先备份到 .bak（一次性，避免每次都覆盖备份）
- 严禁直接展开环境变量 —— 写盘要保留 ``${ENV}`` 占位语法
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

# 优先 ruamel.yaml（保序 + 注释保留）；不可用则降级 PyYAML
try:
    from ruamel.yaml import YAML  # type: ignore

    _RUAMEL = YAML(typ="rt")
    _RUAMEL.preserve_quotes = True
    _RUAMEL.indent(mapping=2, sequence=4, offset=2)
    _USE_RUAMEL = True
except ImportError:  # pragma: no cover
    import yaml as _pyyaml  # type: ignore

    _USE_RUAMEL = False


def load_yaml(path: str | os.PathLike) -> Dict[str, Any]:
    """读取 YAML 文件。文件不存在时返回空 dict，不抛异常。"""
    file_path = Path(path)
    if not file_path.exists():
        logger.warning("load_yaml: %s 不存在，返回空", file_path)
        return {}
    try:
        if _USE_RUAMEL:
            with file_path.open("r", encoding="utf-8") as f:
                data = _RUAMEL.load(f)
            return dict(data) if data else {}
        with file_path.open("r", encoding="utf-8") as f:
            return _pyyaml.safe_load(f) or {}
    except Exception as load_error:
        logger.error("load_yaml: %s 解析失败: %s", file_path, load_error)
        raise


def save_yaml(path: str | os.PathLike, data: Dict[str, Any]) -> None:
    """写回 YAML 文件，首次写入会先备份 ``<file>.bak``。"""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    backup_path = file_path.with_suffix(file_path.suffix + ".bak")
    if file_path.exists() and not backup_path.exists():
        shutil.copy2(file_path, backup_path)
        logger.info("save_yaml: 首次写入，已备份 %s → %s", file_path.name, backup_path.name)

    if _USE_RUAMEL:
        with file_path.open("w", encoding="utf-8") as f:
            _RUAMEL.dump(data, f)
    else:  # pragma: no cover
        with file_path.open("w", encoding="utf-8") as f:
            _pyyaml.safe_dump(  # type: ignore[name-defined]
                data,
                f,
                allow_unicode=True,
                sort_keys=False,
                default_flow_style=False,
            )
    logger.info("save_yaml: 已写入 %s", file_path)
