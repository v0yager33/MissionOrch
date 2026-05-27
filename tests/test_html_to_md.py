"""html_to_md 单元测试：保证 HTML 预处理器对标准 HTML 能输出合理的 Markdown。"""

from __future__ import annotations

from pathlib import Path

import pytest

from missionorch_lc.core.rag.html_to_md import (
    convert_html_file_to_md,
    html_text_to_markdown,
    is_html_file,
)


def test_is_html_file() -> None:
    assert is_html_file(Path("a.html")) is True
    assert is_html_file(Path("a.HTM")) is True
    assert is_html_file(Path("a.xhtml")) is True
    assert is_html_file(Path("a.md")) is False
    assert is_html_file(Path("a.pdf")) is False


def test_headings_and_paragraphs() -> None:
    html = """
    <html><body>
      <h1>SEAD Overview</h1>
      <p>SEAD stands for <strong>Suppression of Enemy Air Defenses</strong>.</p>
      <h2>Phases</h2>
      <p>Three phases: suppress, deceive, strike.</p>
    </body></html>
    """
    md = html_text_to_markdown(html)
    assert "# SEAD Overview" in md
    assert "## Phases" in md
    assert "Suppression of Enemy Air Defenses" in md
    assert "Three phases" in md


def test_lists() -> None:
    html = """
    <body>
      <ul>
        <li>F-16CJ</li>
        <li>EA-18G</li>
      </ul>
      <ol>
        <li>Detect</li>
        <li>Suppress</li>
        <li>Destroy</li>
      </ol>
    </body>
    """
    md = html_text_to_markdown(html)
    assert "- F-16CJ" in md
    assert "- EA-18G" in md
    assert "1. Detect" in md
    assert "2. Suppress" in md
    assert "3. Destroy" in md


def test_table() -> None:
    html = """
    <body>
      <table>
        <tr><th>Aircraft</th><th>Role</th></tr>
        <tr><td>EA-18G</td><td>EW</td></tr>
        <tr><td>F-16CJ</td><td>SEAD</td></tr>
      </table>
    </body>
    """
    md = html_text_to_markdown(html)
    assert "| Aircraft | Role |" in md
    assert "| --- | --- |" in md
    assert "| EA-18G | EW |" in md
    assert "| F-16CJ | SEAD |" in md


def test_code_block_and_link() -> None:
    html = """
    <body>
      <p>See <a href="https://example.com/doc">official doctrine</a>.</p>
      <pre>line1
line2
line3</pre>
    </body>
    """
    md = html_text_to_markdown(html)
    assert "[official doctrine](https://example.com/doc)" in md
    assert "```" in md
    assert "line1" in md and "line3" in md


def test_noise_removed() -> None:
    html = """
    <html><head><style>.x{color:red}</style></head>
    <body>
      <nav>NAV MENU</nav>
      <script>alert('x')</script>
      <main>
        <h1>Real Content</h1>
        <p>Important paragraph.</p>
      </main>
      <footer>FOOTER JUNK</footer>
    </body></html>
    """
    md = html_text_to_markdown(html)
    assert "NAV MENU" not in md
    assert "FOOTER JUNK" not in md
    assert "alert" not in md
    assert "color:red" not in md
    assert "# Real Content" in md
    assert "Important paragraph." in md


def test_no_excessive_blank_lines() -> None:
    html = "<body><p>a</p><p>b</p><p>c</p></body>"
    md = html_text_to_markdown(html)
    # 不应出现连续 3 个空行
    assert "\n\n\n" not in md


def test_convert_html_file_to_md(tmp_path: Path) -> None:
    html_file = tmp_path / "sample.html"
    html_file.write_text(
        "<html><body><h1>Title</h1><p>Body.</p></body></html>",
        encoding="utf-8",
    )
    md_dir = tmp_path / "out"
    md_path = convert_html_file_to_md(html_file, md_dir)
    assert md_path.exists()
    text = md_path.read_text(encoding="utf-8")
    assert "<!-- naive-rag: converted from sample.html" in text
    assert "# Title" in text
    assert "Body." in text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
