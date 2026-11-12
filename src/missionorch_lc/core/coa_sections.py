"""COA Markdown 的章节拆分与补丁合并。

修订轮次里 Planner 只输出需要修改的章节（``revision_mode: section_patch``），
这里按"## X、标题"把补丁合并回上一版全文，未修改的章节逐字保留。
相比整篇重写，输出 token 少得多，也不会出现"改一处、坏三处"。
"""

from __future__ import annotations

import re
from typing import List, Tuple

_H1_RE = re.compile(r"^#\s+COA方案[：:].*$", re.MULTILINE)
_H2_RE = re.compile(r"^##\s+(.+?)\s*$")
_NUM_PREFIX_RE = re.compile(r"^[一二三四五六七八九十\d]+\s*[、.．)）]\s*")
_ALIAS_RE = re.compile(r"[（(][^）)]*[）)]")

Section = Tuple[str, str]  # (标题行, 正文)


def split_sections(text: str) -> Tuple[str, List[Section]]:
    """拆成 (前言, [(标题行, 正文), ...])；前言是第一个 ``## `` 之前的内容。"""
    preamble: List[str] = []
    sections: List[List] = []
    for line in (text or "").splitlines():
        if _H2_RE.match(line):
            sections.append([line.rstrip(), []])
        elif sections:
            sections[-1][1].append(line)
        else:
            preamble.append(line)
    return "\n".join(preamble), [(head, "\n".join(body)) for head, body in sections]


def section_key(heading: str) -> str:
    """"## 六、实体清单（Entity Manifest）" → "实体清单"，用于跨版本匹配章节。"""
    title = heading.lstrip("#").strip()
    title = _NUM_PREFIX_RE.sub("", title)
    title = _ALIAS_RE.sub("", title)
    return re.sub(r"\s+", "", title).lower()


def is_full_document(text: str, min_sections: int = 5) -> bool:
    """带 "# COA方案" 标题且章节数够多，视为完整方案而不是补丁。"""
    if not text or not _H1_RE.search(text):
        return False
    return len(split_sections(text)[1]) >= min_sections


def _find(sections: List[Section], key: str) -> int:
    for idx, (head, _) in enumerate(sections):
        if section_key(head) == key:
            return idx
    for idx, (head, _) in enumerate(sections):
        other = section_key(head)
        if key and other and (key in other or other in key):
            return idx
    return -1


def merge_sections(base: str, patch: str) -> Tuple[str, List[str]]:
    """把 ``patch`` 中的章节替换/追加到 ``base``。

    Returns:
        (合并后的全文, 被替换或追加的章节标题列表)。补丁里没有任何 ``## `` 章节时
        原样返回 ``base`` 和空列表。
    """
    patch_pre, patch_secs = split_sections(patch)
    if not patch_secs:
        return base, []

    base_pre, base_secs = split_sections(base)
    merged = list(base_secs)
    touched: List[str] = []
    for head, body in patch_secs:
        idx = _find(merged, section_key(head))
        if idx >= 0:
            merged[idx] = (merged[idx][0], body)
            touched.append(merged[idx][0].lstrip("# ").strip())
        else:
            merged.append((head, body))
            touched.append(head.lstrip("# ").strip())

    # 补丁里给了新的方案标题时一并替换
    new_title = _H1_RE.search(patch_pre)
    if new_title and _H1_RE.search(base_pre):
        base_pre = _H1_RE.sub(lambda _: new_title.group(0), base_pre, count=1)

    out: List[str] = [base_pre.rstrip()] if base_pre.strip() else []
    for head, body in merged:
        out.append(head)
        out.append(body.rstrip())
        out.append("")
    return "\n".join(out).rstrip() + "\n", touched
