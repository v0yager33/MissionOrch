"""提示词加载工具。

LangChain 的 `ChatPromptTemplate` 默认使用 f-string 风格的 `{var}` 占位符。
我们的 `.txt` 模板里除了少数预定义变量外，还存在大量花括号字面量
（Markdown 表格示例、JSON 示例等），这些 `{`/`}` 会被当作变量名解析而报错。

本模块提供 `escape_prompt_text(text, allowed_vars)` 工具：
- 保留 `allowed_vars` 列表中的占位符（如 `{rag_context}`）
- 将其他所有 `{`、`}` 字面量转义为 `{{`、`}}`
这样既能正常渲染变量，又不会因模板中的花括号片段报错。

以及 `load_prompt_text(path)`：热加载 `.txt` 文件（带 mtime 缓存）。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from threading import Lock
from typing import Dict, Iterable, Tuple

logger = logging.getLogger(__name__)

_cache: Dict[str, Tuple[float, str]] = {}
_cache_lock = Lock()


def load_prompt_text(path: str | Path) -> str:
    """读取 prompt 文件内容，基于 mtime 自动热加载。

    Args:
        path: prompt 文件路径。

    Returns:
        文件原始文本。如果文件不存在返回空字符串。
    """
    if not path:
        return ""
    path_obj = Path(path)
    if not path_obj.exists():
        logger.warning("Prompt file not found: %s", path)
        return ""

    key = str(path_obj.resolve())
    mtime = path_obj.stat().st_mtime

    with _cache_lock:
        cached = _cache.get(key)
        if cached and cached[0] >= mtime:
            return cached[1]

    with open(path_obj, "r", encoding="utf-8") as f:
        content = f.read()

    with _cache_lock:
        _cache[key] = (mtime, content)
    logger.debug("Loaded prompt (%d chars) from %s", len(content), path)
    return content


def escape_prompt_text(text: str, allowed_vars: Iterable[str]) -> str:
    """把 prompt 文本转成 LangChain f-string 模板可安全消费的形式。

    规则：
    1. 先把所有 `{` 替换为 `{{`，所有 `}` 替换为 `}}`（全部转义成字面量）。
    2. 再把 `{{var}}` 模式（原本是 `{var}`）还原为 `{var}` —— 仅对白名单里的变量。

    这样：
    - `{rag_context}` 等已声明变量会被 LangChain 正常解析；
    - `{...}` / `{"k": "v"}` 等 JSON / Markdown 里的花括号会以字面量保留；
    - 不会因未声明的变量名（如 `{name}` 示例）导致 KeyError。

    Args:
        text: 原始 prompt 文本
        allowed_vars: 要保留为变量的名字列表，如 `["rag_context"]`

    Returns:
        转义后的 prompt 文本
    """
    if not text:
        return text

    escaped = text.replace("{", "{{").replace("}", "}}")

    for var in allowed_vars:
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", var):
            raise ValueError(f"Invalid variable name for prompt template: {var!r}")
        # `{var}` 原文被 1 处理成 `{{var}}`，我们恢复为 `{var}`
        pattern = "{{" + var + "}}"
        replacement = "{" + var + "}"
        escaped = escaped.replace(pattern, replacement)

    return escaped


def load_and_escape(path: str | Path, allowed_vars: Iterable[str]) -> str:
    """便捷方法：`load_prompt_text` + `escape_prompt_text`。"""
    text = load_prompt_text(path)
    return escape_prompt_text(text, allowed_vars)
