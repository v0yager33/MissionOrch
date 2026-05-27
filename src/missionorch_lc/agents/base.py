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

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Type

import yaml
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from pydantic import BaseModel

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

    def __init__(self, agent_type: str, config_path: str | None = None) -> None:
        if config_path is None:
            from ..core.settings import get_settings
            config_path = str(get_settings().agents_config)

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

    def bind_structured_output(
        self,
        schema: Any,
        *,
        method: Optional[str] = None,
    ) -> Runnable:
        """对 raw model 调用 `with_structured_output(schema)` 并补上容错包装。

        Args:
            schema: Pydantic 模型类或 JSON schema。
            method: 指定结构化输出方式，可选 "function_calling" / "json_mode" / "json_schema"；
                默认 None 让 LangChain 自行选择。对 DeepSeek 等 reasoning 模型，
                function_calling 可能返回包装过的 AIMessage，建议显式使用 "json_mode"。
        """
        try:
            if method is not None:
                structured = self.raw_model.with_structured_output(schema, method=method)
            else:
                structured = self.raw_model.with_structured_output(schema)
        except NotImplementedError as struct_error:
            raise RuntimeError(
                f"Model '{self.model_id}' does not support structured output: {struct_error}"
            ) from struct_error
        except TypeError:
            # 某些 ChatModel 不接受 method 参数，回退到无参调用
            structured = self.raw_model.with_structured_output(schema)

        runnable: Runnable = structured
        if self.temperature_override is not None:
            runnable = runnable.bind(temperature=self.temperature_override)
        return ModelRouter.wrap_with_resilience(runnable, self.model_id)

    def build_structured_chain_pair(
        self,
        schema: Type[BaseModel],
        prompt: Any,
    ) -> tuple:
        """构建 (structured_chain, fallback_chain) 对。

        Judge 和 Validator 共享的懒加载模式：优先走 with_structured_output（json_mode），
        失败则退回 JsonOutputParser 兜底链。

        Returns:
            (structured_chain_or_None, fallback_chain)
        """
        from langchain_core.output_parsers import JsonOutputParser

        structured_chain = None
        for attempt_method in ("json_mode", None):
            try:
                structured_model = self.bind_structured_output(schema, method=attempt_method)
                structured_chain = prompt | structured_model
                logger.info(
                    f"{self.agent_type}: with_structured_output({schema.__name__}, method={attempt_method})"
                )
                break
            except Exception as build_error:
                logger.debug(
                    f"with_structured_output(method={attempt_method}) failed: {build_error}"
                )
                continue

        if structured_chain is None:
            logger.warning(
                f"{self.agent_type}: with_structured_output unavailable, fallback to JsonOutputParser"
            )

        fallback_chain = prompt | self.bind_model() | JsonOutputParser(pydantic_object=schema)
        return structured_chain, fallback_chain

    async def invoke_structured(
        self,
        schema: Type[BaseModel],
        structured_chain: Any,
        fallback_chain: Any,
        inputs: Dict[str, Any],
        config: Optional[RunnableConfig] = None,
    ) -> Dict[str, Any]:
        """执行 structured chain 并自动回退。

        统一处理：结构化链 → coerce → 核心字段缺失时回退到 fallback 链。
        """
        result_dict: Dict[str, Any]
        try:
            if structured_chain is not None:
                result_obj = await structured_chain.ainvoke(inputs, config=config)
                result_dict = self.coerce_structured_result(result_obj, schema)
                required_fields = set(schema.model_fields.keys())
                has_any = bool(set(result_dict.keys()) & required_fields)
                if not has_any:
                    logger.warning(
                        f"{self.agent_type}: structured result missing core fields, retrying with fallback"
                    )
                    fallback_dict = await fallback_chain.ainvoke(inputs, config=config)
                    if isinstance(fallback_dict, dict):
                        result_dict = fallback_dict
            else:
                result_dict = await fallback_chain.ainvoke(inputs, config=config)
                if not isinstance(result_dict, dict):
                    result_dict = dict(result_dict)
        except Exception as invoke_error:
            logger.error(f"{self.agent_type} structured invoke failed: {invoke_error}")
            raise
        return result_dict

    @staticmethod
    def coerce_structured_result(
        result_obj: Any,
        schema: Optional[Type[BaseModel]] = None,
    ) -> Dict[str, Any]:
        """把 `with_structured_output` 的返回值统一归一化为业务字段 dict。

        处理以下几种实际情况（不同 provider + 不同 method 的返回差异很大）：
          1. Pydantic 模型实例：直接 `model_dump()`。
          2. 已经是业务字段 dict（包含 schema 的必填字段）：原样返回。
          3. LangChain AIMessage 实例（或其 dict 形态）：提取 `content` 里的 JSON 字符串再解析。
          4. 退化：content 不是合法 JSON，返回 `{}`（由调用方再走默认兜底）。
        """
        # 1. AIMessage 等 LangChain 消息实例 —— 注意 BaseMessage 也是 Pydantic BaseModel 的子类，
        #    所以必须先于 BaseModel 判断，否则会把整个消息 dump 成 dict。
        if isinstance(result_obj, BaseMessage):
            return BaseAgent._parse_content_json(result_obj.content)

        # 2. 业务 Pydantic 实例（JudgeResult / ValidationResult 等）
        if isinstance(result_obj, BaseModel):
            return result_obj.model_dump()

        # 3. dict 情况
        if isinstance(result_obj, dict):
            schema_field_names = (
                set(schema.model_fields.keys()) if schema is not None else set()
            )
            # 3a. 业务字段 dict —— 与 schema 字段有交集即认定为业务 dict
            if schema_field_names and (set(result_obj.keys()) & schema_field_names):
                return result_obj
            # 3b. AIMessage dump 出来的 dict（含 content/response_metadata 等）
            if "content" in result_obj and any(
                marker in result_obj
                for marker in ("response_metadata", "additional_kwargs", "type", "tool_calls")
            ):
                return BaseAgent._parse_content_json(result_obj["content"])
            # 3c. 其他 dict：原样返回
            return result_obj

        # 4. 其他未知类型：尽力转 dict
        if hasattr(result_obj, "model_dump"):
            try:
                return result_obj.model_dump()
            except Exception:
                pass
        logger.warning(
            f"coerce_structured_result: unexpected type {type(result_obj).__name__}, returning empty dict"
        )
        return {}

    @staticmethod
    def _parse_content_json(content: Any) -> Dict[str, Any]:
        """从 AIMessage.content（可能是 str 或 list[dict]）中提取并解析 JSON。"""
        if isinstance(content, list):
            # content 是 [{"type": "text", "text": "..."}] 形式时，拼接所有文本
            text_parts = [
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            content_text = "\n".join(text_parts)
        elif isinstance(content, str):
            content_text = content
        else:
            return {}

        if not content_text:
            return {}

        # 优先直接解析
        try:
            parsed = json.loads(content_text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # 兜底：从 ```json ... ``` 代码块或首个花括号对中提取
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content_text, re.DOTALL)
        if fence_match:
            try:
                parsed = json.loads(fence_match.group(1))
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

        brace_match = re.search(r"\{.*\}", content_text, re.DOTALL)
        if brace_match:
            try:
                parsed = json.loads(brace_match.group(0))
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

        return {}

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
