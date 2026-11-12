"""延迟可能属于认证信息的流尾，安全发布原生增量，不预消费业务流。"""
from __future__ import annotations

import copy
import re

from .core import capture_failure, get_recorder

_TEXT = {
    "content", "text", "delta", "thinking", "reasoning_content", "arguments", "partial_json", "input", "unparsed_data",
}


class SafeFrames:
    def __init__(self, publish):
        self.pending = []
        self.publish = publish

    def push(self, event):
        try:
            if get_recorder().enabled:
                self.pending.append(copy.deepcopy(event))
                self.drain(False)
        except Exception:
            self.pending.clear()
            capture_failure()

    def drain(self, final=False):
        try:
            self._drain(final)
        except Exception:
            self.pending.clear()
            capture_failure()

    def _drain(self, final):
        groups = {}
        sanitized = copy.deepcopy(self.pending)

        def walk(value, path=(), field=""):
            if isinstance(value, dict):
                for key, child in value.items():
                    walk(child, (*path, key), key)
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    walk(child, (*path, index), field)
            elif isinstance(value, str) and field in _TEXT:
                # 正文、思考、工具 JSON 分开处理，允许跨 SSE 帧拼接。
                groups.setdefault(field, []).append((path, value))

        walk(sanitized)
        recorder = get_recorder()
        with recorder._lock:
            secrets = tuple(recorder.secrets)
        hold = False
        for leaves in groups.values():
            text = "".join(value for _, value in leaves)
            spans = []
            for secret in secrets:
                spans.extend((m.start(), m.end()) for m in re.finditer(re.escape(secret), text))
            spans.extend( ( m.start( ), m.end( )) for m in re.finditer(
                r"(?i)\b(?:sk-[A-Za-z0-9_-]+|Bearer\s+[A-Za-z0-9._~+/-]+=*)",
                text))
            for prefix in (*secrets, "sk-", "Bearer "):
                for length in range(1, min(len(prefix), len(text) + 1)):
                    if text.endswith(prefix[:length]):
                        if final:
                            spans.append((len(text) - length, len(text)))
                        else:
                            hold = True
            trailing = re.search(r"(?i)(?:sk-[A-Za-z0-9_-]*|Bearer\s+[A-Za-z0-9._~+/-]*=*)$", text)
            if trailing:
                if final:
                    spans.append(trailing.span())
                else:
                    hold = True
            offset = 0
            for path, value in leaves:
                chars = list(value)
                for start, end in spans:
                    lo, hi = max(start - offset, 0), min(end - offset, len(chars))
                    if lo < hi:
                        # 固定长度替换避免位置漂移，任何片段都不能重组凭据。
                        chars[lo:hi] = "*" * (hi - lo)
                target = sanitized
                for part in path[:-1]:
                    target = target[part]
                target[path[-1]] = "".join(chars)
                offset += len(value)
        if hold:
            return
        frames, self.pending = sanitized, []
        for frame in frames:
            self.publish(frame)
