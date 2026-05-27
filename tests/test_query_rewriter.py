"""MultiQueryRewriter 的纯函数单元测试。

只测 ``_parse_queries``（静态方法，不依赖 LLM / 网络），保证：
- 纯 JSON 能被正确解析
- 带 ```json ... ``` / ``` ... ``` 包裹能被去包裹
- JSON 前后带解释文字也能抽出来
- queries 不是 list 或缺失时返回 []
- 空字符串 / None-like 返回 []
- queries 内包含非字符串 / 空字符串时会被过滤
"""

from __future__ import annotations

from missionorch_lc.core.rag.query_rewriter import MultiQueryRewriter


class TestParseQueries:
    """覆盖 ``_parse_queries`` 所有分支。"""

    def test_plain_json(self) -> None:
        raw = '{"queries": ["a", "b", "c"]}'
        assert MultiQueryRewriter._parse_queries(raw) == ["a", "b", "c"]

    def test_json_with_whitespace(self) -> None:
        raw = '   {"queries": ["x"]}  \n  '
        assert MultiQueryRewriter._parse_queries(raw) == ["x"]

    def test_markdown_json_fence(self) -> None:
        raw = '```json\n{"queries": ["m1", "m2"]}\n```'
        assert MultiQueryRewriter._parse_queries(raw) == ["m1", "m2"]

    def test_generic_code_fence(self) -> None:
        raw = '```\n{"queries": ["g1"]}\n```'
        assert MultiQueryRewriter._parse_queries(raw) == ["g1"]

    def test_leading_explanation_before_json(self) -> None:
        # LLM 有时会啰嗦："好的，这是结果：{...}"
        raw = '好的，这是结果：\n{"queries": ["q1", "q2"]}'
        assert MultiQueryRewriter._parse_queries(raw) == ["q1", "q2"]

    def test_trailing_text_after_json(self) -> None:
        raw = '{"queries": ["tail1"]}\n（解释：...）'
        assert MultiQueryRewriter._parse_queries(raw) == ["tail1"]

    def test_empty_string(self) -> None:
        assert MultiQueryRewriter._parse_queries("") == []

    def test_none_like_input(self) -> None:
        # 传空字符串应返回 []（类型由调用方保证是 str）
        assert MultiQueryRewriter._parse_queries("   ") == []

    def test_queries_missing(self) -> None:
        # 合法 JSON 但没有 queries key → []
        raw = '{"other_key": ["a"]}'
        assert MultiQueryRewriter._parse_queries(raw) == []

    def test_queries_not_a_list(self) -> None:
        # queries 是字符串而不是 list → []
        raw = '{"queries": "not-a-list"}'
        assert MultiQueryRewriter._parse_queries(raw) == []

    def test_filters_empty_and_whitespace_items(self) -> None:
        raw = '{"queries": ["valid", "", "   ", "another"]}'
        assert MultiQueryRewriter._parse_queries(raw) == ["valid", "another"]

    def test_non_string_items_coerced(self) -> None:
        # 数字 / null 也要稳健处理
        raw = '{"queries": ["s1", 42, null, "s2"]}'
        assert MultiQueryRewriter._parse_queries(raw) == ["s1", "42", "s2"]

    def test_not_json_at_all(self) -> None:
        # 完全不是 JSON（之前有按行兜底，现在明确返回 [] 给调用方降级）
        raw = "just some prose without any json"
        assert MultiQueryRewriter._parse_queries(raw) == []

    def test_nested_json_inside(self) -> None:
        # JSONDecoder.raw_decode 能正确处理含嵌套结构
        raw = (
            '{"queries": ["a"], "meta": {"source": "test"}}\n'
            "剩余无关文字 }}}"
        )
        assert MultiQueryRewriter._parse_queries(raw) == ["a"]
