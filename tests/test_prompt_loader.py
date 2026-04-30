"""prompt_loader 单元测试。"""

import tempfile
from pathlib import Path

from missionorch_lc.core.prompt_loader import (
    escape_prompt_text,
    load_and_escape,
    load_prompt_text,
)


class TestEscapePromptText:
    """escape_prompt_text 白名单转义逻辑。"""

    def test_empty_string(self):
        assert escape_prompt_text("", ["var"]) == ""

    def test_no_braces(self):
        text = "Hello world, no braces here."
        assert escape_prompt_text(text, []) == text

    def test_all_braces_escaped_when_no_allowed_vars(self):
        text = "Use {name} and {age} in template."
        result = escape_prompt_text(text, [])
        assert result == "Use {{name}} and {{age}} in template."

    def test_allowed_vars_preserved(self):
        text = "Hello {rag_context}, see {other_thing}."
        result = escape_prompt_text(text, ["rag_context"])
        assert "{rag_context}" in result
        assert "{{other_thing}}" in result

    def test_json_example_escaped(self):
        text = '示例: {"key": "value", "score": 8.5}'
        result = escape_prompt_text(text, [])
        assert "{" not in result.replace("{{", "").replace("}}", "")

    def test_multiple_allowed_vars(self):
        text = "COA: {coa_table}, Mission: {mission}, Criteria: {criteria}, JSON: {\"a\": 1}"
        result = escape_prompt_text(text, ["coa_table", "mission", "criteria"])
        assert "{coa_table}" in result
        assert "{mission}" in result
        assert "{criteria}" in result
        # JSON 部分的花括号被转义
        assert '{{' in result

    def test_nested_braces(self):
        text = "{{already_escaped}} and {var}"
        result = escape_prompt_text(text, ["var"])
        # 原来的 {{ 变成 {{{{ ，然后 {var} 保留
        assert "{var}" in result

    def test_validator_prompt_pattern(self):
        """模拟 validator.txt 中的 JSON 示例。"""
        text = """验证规则:
{validation_rules}

示例输出:
  "corrected_coa": {...},
  "ARRIVE_TARGET": [{{"task_id": "PATROL_A"}}]
任务: {mission}"""
        result = escape_prompt_text(text, ["validation_rules", "mission"])
        assert "{validation_rules}" in result
        assert "{mission}" in result
        # `{...}` 被转义
        assert "{{...}}" in result


class TestLoadPromptText:
    """load_prompt_text 文件读取与缓存。"""

    def test_nonexistent_file(self):
        result = load_prompt_text("/nonexistent/path/prompt.txt")
        assert result == ""

    def test_empty_path(self):
        result = load_prompt_text("")
        assert result == ""

    def test_reads_file(self, tmp_path: Path):
        prompt_file = tmp_path / "test.txt"
        prompt_file.write_text("Hello {name}", encoding="utf-8")
        result = load_prompt_text(str(prompt_file))
        assert result == "Hello {name}"

    def test_cache_reuses(self, tmp_path: Path):
        prompt_file = tmp_path / "cached.txt"
        prompt_file.write_text("version1", encoding="utf-8")

        result1 = load_prompt_text(str(prompt_file))
        assert result1 == "version1"

        # 不修改文件 → 从缓存读取
        result2 = load_prompt_text(str(prompt_file))
        assert result2 == "version1"


class TestLoadAndEscape:
    """load_and_escape 便捷方法。"""

    def test_load_and_escape(self, tmp_path: Path):
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text(
            "Context: {rag_context}\nJSON: {\"key\": 1}", encoding="utf-8"
        )
        result = load_and_escape(str(prompt_file), ["rag_context"])
        assert "{rag_context}" in result
        assert "{{" in result  # JSON 花括号被转义
