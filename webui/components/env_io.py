"""项目根目录 .env 文件读写工具。

简单的 key=value 行格式（兼容 dotenv）：
    OPENAI_API_KEY=sk-xxx
    DEEPSEEK_API_KEY=sk-yyy
    # 注释行原样保留

写回时只更新已有的 key，未声明的 key 追加到文件末尾，**保留注释**。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


# 已知的 LLM Provider 环境变量（前端默认渲染这些 key）
KNOWN_API_KEYS: List[str] = [
    "OPENAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "ARK_API_KEY",
    "DASHSCOPE_API_KEY",
    "LANGSMITH_API_KEY",
    "LANGSMITH_PROJECT",
]


def load_env(env_path: str | os.PathLike) -> Dict[str, str]:
    """读取 .env 为 dict（忽略注释 / 空行）。"""
    file_path = Path(env_path)
    if not file_path.exists():
        return {}
    pairs: Dict[str, str] = {}
    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        pairs[key.strip()] = value.strip().strip('"').strip("'")
    return pairs


def save_env(env_path: str | os.PathLike, updates: Dict[str, str]) -> None:
    """更新 .env 文件，**保留原有注释与顺序**，未声明的 key 追加到末尾。"""
    file_path = Path(env_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    existing_lines: List[str] = []
    if file_path.exists():
        existing_lines = file_path.read_text(encoding="utf-8").splitlines()

    seen_keys: set[str] = set()
    new_lines: List[str] = []
    for raw_line in existing_lines:
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#") or "=" not in line:
            new_lines.append(line)
            continue
        key, _, _old_value = line.partition("=")
        key = key.strip()
        if key in updates:
            new_lines.append(f"{key}={updates[key]}")
            seen_keys.add(key)
        else:
            new_lines.append(line)

    # 追加首次出现的 key
    for key, value in updates.items():
        if key in seen_keys:
            continue
        new_lines.append(f"{key}={value}")

    file_path.write_text("\n".join(new_lines).rstrip() + "\n", encoding="utf-8")
    logger.info("save_env: 已写入 %s（更新 %d 个 key）", file_path, len(updates))


def merge_env_into_process(env_pairs: Dict[str, str]) -> None:
    """把 .env 里的 key=value 注入当前进程，让本次会话生效。"""
    for key, value in env_pairs.items():
        if value:
            os.environ[key] = value
