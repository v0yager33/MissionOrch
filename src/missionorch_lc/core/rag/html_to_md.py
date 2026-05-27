"""HTML → Markdown 轻量转换器（朴素 RAG 索引预处理）。

为什么需要这层？
    RAGAnything 的 ``parser=mineru`` (pipeline backend) 对 HTML 没有原生
    分支（源码 `parser.py:1285-1313`），遇到 HTML 会走 else "try as PDF"
    导致失败。而我们又不想切到 docling（依赖更重、已索引的 doctrines 要重建）。

最轻改造：
    索引脚本遇到 .html / .htm / .xhtml 时，先用本模块把它转成 Markdown
    （bs4 剥标签 + 保留基本结构：标题、段落、列表、表格、链接），
    存到一个临时 .md 文件，再交给 MinerU 的 TEXT_FORMATS 分支处理。

副作用：
    - 复杂表格的跨行跨列信息会丢（只保留「文本对齐 + 分隔线」的朴素 md 表格）
    - `<script>`、`<style>`、`<nav>`、`<footer>` 等噪声节点会被剔除
    - 对于「作战条例/战例文本」这种以纯文本为主的 HTML，效果足够
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# 需要彻底剔除的节点（噪声 / 脚本 / 样式）
_NOISE_TAGS = {
    "script",
    "style",
    "noscript",
    "iframe",
    "nav",
    "footer",
    "header",
    "aside",
    "form",
    "svg",
    "button",
}

# 标题级别映射
_HEADING_LEVELS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}


def html_text_to_markdown(html: str) -> str:
    """核心实现：HTML 字符串 → Markdown 字符串。

    - 使用 bs4 解析 DOM
    - 剔除 _NOISE_TAGS
    - 按 DOM 顺序 walk，生成 md 片段
    - 标题、段落、列表、表格、链接、代码块都做最低限度保留
    """
    from bs4 import BeautifulSoup, NavigableString, Tag  # 延迟 import

    soup = BeautifulSoup(html, "html.parser")

    # 先剔除噪声节点
    for tag_name in _NOISE_TAGS:
        for noise in soup.find_all(tag_name):
            noise.decompose()

    # 优先找 <main> / <article>；没有就用 <body>；再没有就整个 soup
    root: Optional[Tag] = (
        soup.find("main") or soup.find("article") or soup.body or soup
    )

    lines: List[str] = []

    def _walk(node) -> None:
        if isinstance(node, NavigableString):
            text = str(node).strip()
            if text:
                lines.append(text)
            return

        if not isinstance(node, Tag):
            return

        name = node.name.lower()

        # 标题
        if name in _HEADING_LEVELS:
            level = _HEADING_LEVELS[name]
            text = _render_inline(node).strip()
            if text:
                lines.append("")
                lines.append(f"{'#' * level} {text}")
                lines.append("")
            return

        # 段落：走行内渲染器，保留 <a>/<code>/<strong>/<em> 结构
        if name == "p":
            text = _render_inline(node).strip()
            if text:
                lines.append("")
                lines.append(text)
                lines.append("")
            return

        # 水平分割线
        if name == "hr":
            lines.append("")
            lines.append("---")
            lines.append("")
            return

        # 换行
        if name == "br":
            lines.append("")
            return

        # 无序列表
        if name == "ul":
            lines.append("")
            for li in node.find_all("li", recursive=False):
                text = _render_inline(li).strip()
                if text:
                    lines.append(f"- {text}")
            lines.append("")
            return

        # 有序列表
        if name == "ol":
            lines.append("")
            for idx, li in enumerate(node.find_all("li", recursive=False), start=1):
                text = _render_inline(li).strip()
                if text:
                    lines.append(f"{idx}. {text}")
            lines.append("")
            return

        # 表格 → 朴素 md 表格（列宽对齐交给 md，跨行跨列丢失）
        if name == "table":
            rows = _extract_table_rows(node)
            if rows:
                lines.append("")
                lines.extend(_table_rows_to_md(rows))
                lines.append("")
            return

        # 代码块
        if name == "pre":
            text = node.get_text("\n", strip=False).rstrip()
            if text:
                lines.append("")
                lines.append("```")
                lines.append(text)
                lines.append("```")
                lines.append("")
            return

        # 块级 <a>（脱离 <p>/<li> 语境的少见情况）
        if name == "a":
            rendered = _render_inline(node).strip()
            if rendered:
                lines.append(rendered)
            return

        # 块级 <img>
        if name == "img":
            rendered = _render_inline(node).strip()
            if rendered:
                lines.append(rendered)
            return

        # 默认：递归子节点
        for child in node.children:
            _walk(child)

    _walk(root)

    # 合并 + 清掉重复空行
    md = "\n".join(lines)
    compact: List[str] = []
    prev_blank = False
    for raw_line in md.splitlines():
        line = raw_line.rstrip()
        is_blank = not line.strip()
        if is_blank and prev_blank:
            continue
        compact.append(line)
        prev_blank = is_blank
    return "\n".join(compact).strip() + "\n"


def _render_inline(node) -> str:
    """把一个『文本容器』节点（<p>/<li>/<h*> 等）渲染为**单行 Markdown 文本**，
    递归保留行内结构：链接、行内代码、加粗、斜体、图片。

    不处理块级标签（<p> / <ul> / <table> 等）—— 那是 _walk 的职责。
    """
    from bs4 import NavigableString, Tag  # 延迟 import

    parts: List[str] = []

    def _go(n) -> None:
        if isinstance(n, NavigableString):
            text = str(n)
            # 不 strip 中间空格，避免把 "see <a>link</a>." 压成 "seelink."
            parts.append(text)
            return
        if not isinstance(n, Tag):
            return

        tag = n.name.lower()

        # 行内链接：[text](href)
        if tag == "a":
            inner = "".join(_render_inline_child_to_str(c) for c in n.children).strip()
            href = n.get("href") or ""
            if inner and href:
                parts.append(f"[{inner}]({href})")
            elif inner:
                parts.append(inner)
            return

        # 行内图片：![alt](src)
        if tag == "img":
            alt = n.get("alt") or ""
            src = n.get("src") or ""
            if alt or src:
                parts.append(f"![{alt}]({src})")
            return

        # 加粗
        if tag in ("strong", "b"):
            inner = "".join(_render_inline_child_to_str(c) for c in n.children).strip()
            if inner:
                parts.append(f"**{inner}**")
            return

        # 斜体
        if tag in ("em", "i"):
            inner = "".join(_render_inline_child_to_str(c) for c in n.children).strip()
            if inner:
                parts.append(f"*{inner}*")
            return

        # 行内代码
        if tag == "code":
            inner = n.get_text("", strip=False)
            if inner:
                parts.append(f"`{inner}`")
            return

        # 软换行在行内压成空格
        if tag == "br":
            parts.append(" ")
            return

        # 其它行内标签（<span> / <u> / <small> / <sub> / <sup> …）
        # 直接递归其子节点
        for c in n.children:
            _go(c)

    def _render_inline_child_to_str(child) -> str:
        """子节点渲染到一个 **字符串** 而不是追加到外层 parts，
        方便 <a>/<strong>/<em> 拿到内部文本。"""
        buf: List[str] = []

        def _local(n) -> None:
            if isinstance(n, NavigableString):
                buf.append(str(n))
                return
            if not isinstance(n, Tag):
                return
            # 行内嵌套先解开子节点文本即可，不再递归支持再一层标记
            buf.append(n.get_text("", strip=False))

        _local(child)
        return "".join(buf)

    for child in node.children:
        _go(child)

    # 压多空格为单空格
    raw = "".join(parts)
    return " ".join(raw.split())


def _extract_table_rows(table_tag) -> List[List[str]]:
    """把 <table> 节点拍成二维列表（每个单元格是 strip 后的 text）。"""
    rows: List[List[str]] = []
    for tr in table_tag.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        if not cells:
            continue
        row = [c.get_text(" ", strip=True) for c in cells]
        rows.append(row)
    return rows


def _table_rows_to_md(rows: List[List[str]]) -> List[str]:
    """二维列表 → md 表格（第一行当表头；把所有行补齐到相同列数）。"""
    if not rows:
        return []
    max_cols = max(len(r) for r in rows)
    padded = [r + [""] * (max_cols - len(r)) for r in rows]
    header = padded[0]
    body = padded[1:]
    out: List[str] = []
    out.append("| " + " | ".join(header) + " |")
    out.append("| " + " | ".join(["---"] * max_cols) + " |")
    for row in body:
        out.append("| " + " | ".join(row) + " |")
    return out


def convert_html_file_to_md(
    html_path: Path,
    md_output_dir: Path,
) -> Path:
    """读一个 .html/.htm/.xhtml 文件，生成同名 .md，返回 md 路径。

    Args:
        html_path: HTML 源文件
        md_output_dir: MD 输出目录（会自动创建）

    Returns:
        生成的 md 文件路径
    """
    md_output_dir.mkdir(parents=True, exist_ok=True)
    md_path = md_output_dir / f"{html_path.stem}.md"

    # 用 errors='ignore' 兼容少量坏编码（军事公开资料里常见 ISO-8859-1 残片）
    raw = html_path.read_text(encoding="utf-8", errors="ignore")
    md = html_text_to_markdown(raw)

    # 文件头加一条溯源注释，方便以后排查
    header = (
        f"<!-- naive-rag: converted from {html_path.name} by html_to_md -->\n\n"
    )
    md_path.write_text(header + md, encoding="utf-8")
    logger.info(
        "html_to_md: %s (%d bytes) → %s (%d bytes)",
        html_path.name,
        len(raw),
        md_path.name,
        md_path.stat().st_size,
    )
    return md_path


def is_html_file(path: Path) -> bool:
    """判断文件扩展名是不是需要走 html→md 预处理的类型。"""
    return path.suffix.lower() in {".html", ".htm", ".xhtml"}
