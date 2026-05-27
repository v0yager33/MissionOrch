"""pytest 公共 fixture / 路径注入。

使 `pytest` 可以直接发现 `src/` 下的 ``missionorch_lc`` 包，无需 `pip install -e .`。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
