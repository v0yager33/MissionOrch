"""MissionOrch-LC: LangChain-based multi-agent COA orchestration."""

__version__ = "2.1.0"

import os as _os
import sys as _sys


def _import_langchain_core_without_transformers() -> None:
    """langchain_core.language_models.base 在模块顶层可选导入 transformers，
    只为了 GPT2 tokenizer 这个兜底计数器（ChatOpenAI 自己用 tiktoken，本项目用不到）。
    装了 transformers 的环境里这一步在 NAS 上要 ~15s。导入期间临时屏蔽，
    导完即恢复，RAG 本地模型仍可正常 import transformers。
    MISSIONORCH_EAGER_TRANSFORMERS=1 可关闭。
    """
    if _os.environ.get("MISSIONORCH_EAGER_TRANSFORMERS") == "1":
        return
    if "transformers" in _sys.modules or "langchain_core.language_models.base" in _sys.modules:
        return
    _sys.modules["transformers"] = None  # type: ignore[assignment]
    try:
        import langchain_core.language_models.base  # noqa: F401
    except ImportError:
        pass
    finally:
        if _sys.modules.get("transformers", 0) is None:
            del _sys.modules["transformers"]


_import_langchain_core_without_transformers()

from .orchestrator import COAOrchestrator  # noqa: E402
from .orchestrator_graph import build_graph, run_graph  # noqa: E402
from .schemas.coa import COA  # noqa: E402
from .schemas.result import COAResult  # noqa: E402

__all__ = [
    "COAOrchestrator",
    "build_graph",
    "run_graph",
    "COA",
    "COAResult",
]
