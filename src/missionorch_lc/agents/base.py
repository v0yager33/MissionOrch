"""Agent 基类 —— 统一的配置加载、模型绑定、提示词热加载。

每个 Agent 的核心是一条 LCEL Chain：
    ChatPromptTemplate | ChatModel | OutputParser

子类职责：
- 声明本 Agent 需要的提示词变量（`prompt_variables`）
- 组装 ChatPromptTemplate（含 system prompt + user message 模板）
- 选择合适的 OutputParser / 或使用 `with_structured_output`
- 暴露业务方法（如 generate_coa / evaluate / reflect / validate）
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool

from ..core.model_router import ModelRouter
from ..core.prompt_loader import escape_prompt_text, load_prompt_text

logger = logging.getLogger(__name__)
interaction_logger = logging.getLogger("missionorch_lc.interactions")


class AgentConfigError(Exception):
    """Agent 配置错误。"""


class BaseAgent:
    """所有 LangChain Agent 的公共基类。"""

    # 子类覆盖：system prompt 中允许被 LangChain 模板解析的变量名白名单
    prompt_variables: Iterable[str] = ()

    def __init__(self, agent_type: str, config_path: str = "config/agents.yaml") -> None:
        self.agent_type = agent_type
        self.config_path = config_path

        self.config = self._load_config(config_path)
        self.agent_cfg = self._validate_agent_config(self.config, agent_type)
        self.workflow_cfg = self.config.get("workflow", {})

        self.model_id: str = self.agent_cfg["model_id"]
        # 原始 ChatModel（用于 bind / bind_tools / with_structured_output 等需要具体类型的方法）
        self.raw_model: BaseChatModel = ModelRouter.get_raw(self.model_id)

        # 温度覆盖：使用 LangChain 的 .bind 机制注入到请求
        self.temperature_override: Optional[float] = self.agent_cfg.get("temperature_override")

        # RAG 开关（实际检索由 RAG Tool 负责，这里只保留开关状态）
        self.use_rag: bool = bool(self.agent_cfg.get("use_rag", False))
        self.rag_retrieval_depth: int = int(self.agent_cfg.get("rag_retrieval_depth", 5))

        # 提示词路径 —— 实际加载/转义走 prompt_loader（带 mtime 缓存）
        self.prompt_path: Optional[str] = self.agent_cfg.get("prompt_file")

        logger.info(f"BaseAgent '{agent_type}' initialized with model '{self.model_id}'")

    # ── 配置加载 ──
    def _load_config(self, path: str) -> Dict[str, Any]:
        config_path = Path(path).resolve()
        if not config_path.exists():
            raise AgentConfigError(f"Agent 配置文件不存在: {path}")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as yaml_error:
            raise AgentConfigError(f"YAML 解析错误: {yaml_error}") from yaml_error

        if not isinstance(config, dict) or "agents" not in config:
            raise AgentConfigError(f"Agent 配置文件格式错误: {path}")
        return config

    def _validate_agent_config(self, config: Dict[str, Any], agent_type: str) -> Dict[str, Any]:
        if agent_type not in config["agents"]:
            available = list(config["agents"].keys())
            raise AgentConfigError(f"未知 agent 类型 '{agent_type}'，可用: {available}")
        agent_cfg = config["agents"][agent_type]
        if "model_id" not in agent_cfg:
            raise AgentConfigError(f"Agent '{agent_type}' 缺少 model_id")
        return agent_cfg

    # ── 提示词热加载（带白名单转义） ──
    @property
    def system_prompt_raw(self) -> str:
        """未经模板转义的原始 prompt 文本（便于日志展示）。"""
        if not self.prompt_path:
            return ""
        return load_prompt_text(self.prompt_path)

    @property
    def system_prompt(self) -> str:
        """经过转义、可被 LangChain f-string 模板安全消费的 system prompt。"""
        return escape_prompt_text(self.system_prompt_raw, self.prompt_variables)

    # ── 模型绑定工具 / 温度 / 容错 ──
    def bind_model(
        self,
        tools: Optional[List[BaseTool]] = None,
        *,
        extra_bind: Optional[Dict[str, Any]] = None,
    ) -> Runnable:
        """基于 raw model 重新构建：bind(temperature) → bind_tools → with_retry → with_fallbacks。

        说明：ModelRouter 缓存了已经包装好的 Runnable（`.with_retry().with_fallbacks()`），
        但 `RunnableWithFallbacks` 不支持 `.bind_tools()` / `.with_structured_output()`。
        所以这里显式基于 raw model 重新走一遍「业务绑定 → 容错包装」链路，
        保证 tool-calling / 温度覆盖 / 重试 / 降级都能共存。
        """
        runnable: Runnable = self.raw_model

        if self.temperature_override is not None:
            runnable = runnable.bind(temperature=self.temperature_override)

        if extra_bind:
            runnable = runnable.bind(**extra_bind)

        if tools:
            # bind_tools 必须在 raw ChatModel 上调用；某些 provider 不支持时会抛错
            try:
                runnable = self.raw_model.bind_tools(tools)
                if self.temperature_override is not None:
                    runnable = runnable.bind(temperature=self.temperature_override)
                if extra_bind:
                    runnable = runnable.bind(**extra_bind)
            except NotImplementedError as tool_error:
                raise RuntimeError(
                    f"Model '{self.model_id}' does not support tool calling: {tool_error}"
                ) from tool_error

        return ModelRouter.wrap_with_resilience(runnable, self.model_id)

    def bind_structured_output(self, schema: Any) -> Runnable:
        """对 raw model 调用 `with_structured_output(schema)` 并补上容错包装。"""
        runnable: Runnable = self.raw_model
        if self.temperature_override is not None:
            runnable = runnable.bind(temperature=self.temperature_override)
        try:
            runnable = self.raw_model.with_structured_output(schema)
            if self.temperature_override is not None:
                runnable = runnable.bind(temperature=self.temperature_override)
        except NotImplementedError as struct_error:
            raise RuntimeError(
                f"Model '{self.model_id}' does not support structured output: {struct_error}"
            ) from struct_error
        return ModelRouter.wrap_with_resilience(runnable, self.model_id)

    # ── 交互日志辅助 ──
    def log_interaction(self, stage: str, prompt_text: str, user_content: str, response: str) -> None:
        separator = "=" * 80
        interaction_logger.info(
            f"\n{separator}\n"
            f"[AGENT: {self.agent_type}] [STAGE: {stage}] [MODEL: {self.model_id}]\n"
            f"{separator}\n"
            f"--- SYSTEM PROMPT ({len(prompt_text)} chars) ---\n{prompt_text}\n"
            f"--- USER CONTENT ({len(user_content)} chars) ---\n{user_content}\n"
            f"--- RESPONSE ({len(response)} chars) ---\n{response}\n"
            f"{separator}"
        )

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """支持 `a.b.c` 嵌套 key 访问。"""
        value: Any = self.agent_cfg
        for k in key.split("."):
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    # ── RunnableConfig 合并 ──
    def merge_config(
        self, config: Optional[RunnableConfig], *, run_name: str
    ) -> RunnableConfig:
        """把默认的 tags / run_name 合并到调用方传入的 RunnableConfig。"""
        merged: Dict[str, Any] = dict(config or {})
        tags = list(merged.get("tags") or [])
        tags.extend([f"agent:{self.agent_type}", f"model:{self.model_id}"])
        merged["tags"] = tags
        merged.setdefault("run_name", run_name)
        return merged  # type: ignore[return-value]