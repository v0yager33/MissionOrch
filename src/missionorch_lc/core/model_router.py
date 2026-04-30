"""ModelRouter：基于 LangChain ChatModel 的统一模型路由（带重试与降级）。

将原有的自研 Adapter 全部替换为 LangChain 官方 ChatModel：
  - openai             -> `ChatOpenAI`
  - openai_compatible  -> `ChatOpenAI(base_url=...)`（豆包 / DeepSeek / Qwen / 本地 Ollama 等）
  - doubao             -> `ChatOpenAI(base_url=...)`（豆包官方使用 OpenAI 兼容协议）
  - anthropic          -> `ChatAnthropic`
  - gemini             -> `ChatGoogleGenerativeAI`

所有模型实例：
  - 单例缓存（按 model_id）
  - 自动包裹 `.with_retry(...)`：指数退避 + 多次重试
  - 可选 `.with_fallbacks([...])`：在主模型失败时降级到备用模型
  - 支持 YAML 中 `${ENV_VAR}` / `${ENV_VAR:default}` 的环境变量占位语法

在模型 YAML 中新增的可选字段：
  max_retries: int  (默认 3)
  fallback_model_ids: list[str]  (按顺序依次降级)
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

logger = logging.getLogger(__name__)


class ModelRouterError(Exception):
    """模型路由器异常基类。"""


class ConfigLoadError(ModelRouterError):
    """配置加载错误。"""


class UnknownProviderError(ModelRouterError):
    """未知 provider 错误。"""


class ModelRouter:
    """LangChain ChatModel 工厂 + 单例缓存。"""

    # 原始 ChatModel 实例（未包装）
    _raw_instances: Dict[str, BaseChatModel] = {}
    # 包装后的 Runnable（with_retry / with_fallbacks）
    _wrapped_instances: Dict[str, Runnable] = {}
    _config_cache: Dict[str, tuple] = {}
    _config_cache_ttl = 300

    # ── 对外主入口 ──
    @classmethod
    def get(
        cls,
        model_id: str,
        config_path: str = "config/models.yaml",
        *,
        wrapped: bool = True,
    ) -> BaseChatModel:
        """按 model_id 获取 ChatModel。

        Args:
            model_id: 模型标识
            config_path: YAML 配置路径
            wrapped: 是否返回带重试/降级包装的 Runnable
                - True（默认）：适合直接用于业务（大多数场景）
                - False：返回原始 ChatModel，便于调用 `bind`/`bind_tools`/`with_structured_output` 等需要具体类型的方法
        """
        cache = cls._wrapped_instances if wrapped else cls._raw_instances
        if model_id in cache:
            return cache[model_id]  # type: ignore[return-value]

        config = cls._load_config_cached(config_path)
        if "models" not in config:
            raise ConfigLoadError(f"配置中缺少 'models' 键: {config_path}")

        if model_id not in config["models"]:
            available = list(config["models"].keys())
            raise KeyError(f"模型 '{model_id}' 不存在。可用模型: {available}")

        model_cfg = config["models"][model_id]
        raw_model = cls._raw_instances.get(model_id) or cls._create_chat_model(model_cfg)
        cls._raw_instances[model_id] = raw_model
        logger.info(f"Created ChatModel '{model_id}' (provider={model_cfg.get('provider')})")

        if not wrapped:
            return raw_model

        wrapped_model = cls._wrap_with_resilience(
            raw_model, model_cfg, config_path, exclude_model_id=model_id
        )
        cls._wrapped_instances[model_id] = wrapped_model
        return wrapped_model  # type: ignore[return-value]

    @classmethod
    def clear_cache(cls) -> None:
        """清空所有缓存。"""
        cls._raw_instances.clear()
        cls._wrapped_instances.clear()
        cls._config_cache.clear()

    # ── 容错包装（公开 API） ──
    @classmethod
    def wrap_with_resilience(
        cls,
        runnable: Runnable,
        model_id: str,
        config_path: str = "config/models.yaml",
    ) -> Runnable:
        """按 `model_id` 的 YAML 配置给任意 Runnable 套上 with_retry + with_fallbacks。

        Args:
            runnable: 任意 LangChain Runnable（常见：`raw_model.bind(...)`、`raw_model.bind_tools(...)`、
                `raw_model.with_structured_output(...)` 的结果）。
            model_id: 用于读取 `max_retries` / `fallback_model_ids` 的模型 ID。
            config_path: models.yaml 路径。
        """
        try:
            cfg = cls._load_config_cached(config_path)["models"][model_id]
        except KeyError as cfg_error:
            logger.warning(
                f"wrap_with_resilience: model_id '{model_id}' not found in {config_path}: {cfg_error}"
            )
            return runnable
        return cls._wrap_with_resilience(runnable, cfg, config_path, exclude_model_id=model_id)

    # ── 容错包装（内部实现） ──
    @classmethod
    def _wrap_with_resilience(
        cls,
        model: Runnable,
        cfg: Dict[str, Any],
        config_path: str,
        *,
        exclude_model_id: str,
    ) -> Runnable:
        """给 Runnable 套一层 with_retry（+ with_fallbacks 如果配置了）。"""
        max_retries = int(cfg.get("max_retries", 3))

        wrapped: Runnable = model
        if max_retries > 0:
            wrapped = wrapped.with_retry(
                stop_after_attempt=max_retries,
                wait_exponential_jitter=True,
                retry_if_exception_type=(Exception,),
            )

        fallback_ids: List[str] = list(cfg.get("fallback_model_ids") or [])
        fallback_ids = [m for m in fallback_ids if m != exclude_model_id]
        if fallback_ids:
            fallbacks: List[Runnable] = []
            for fb_id in fallback_ids:
                try:
                    fb_cfg = cls._load_config_cached(config_path)["models"][fb_id]
                    fb_raw = cls._raw_instances.get(fb_id) or cls._create_chat_model(fb_cfg)
                    cls._raw_instances[fb_id] = fb_raw
                    fb_max_retries = int(fb_cfg.get("max_retries", 2))
                    fb_runnable: Runnable = fb_raw
                    if fb_max_retries > 0:
                        fb_runnable = fb_runnable.with_retry(
                            stop_after_attempt=fb_max_retries,
                            wait_exponential_jitter=True,
                        )
                    fallbacks.append(fb_runnable)
                    logger.info(f"Registered fallback model: {fb_id}")
                except Exception as fb_error:
                    logger.warning(f"Skip fallback '{fb_id}': {fb_error}")
            if fallbacks:
                wrapped = wrapped.with_fallbacks(fallbacks)
        return wrapped

    @classmethod
    def get_raw(cls, model_id: str, config_path: str = "config/models.yaml") -> BaseChatModel:
        """返回原始 ChatModel（未包装），用于 bind/bind_tools/with_structured_output。"""
        return cls.get(model_id, config_path=config_path, wrapped=False)  # type: ignore[return-value]

    # ── ChatModel 构造 ──
    @staticmethod
    def _create_chat_model(cfg: Dict[str, Any]) -> BaseChatModel:
        provider = cfg.get("provider")
        if not provider:
            raise ValueError("配置中缺少 'provider' 字段")

        model = cfg["model"]
        temperature = cfg.get("default_temperature", 0.7)
        max_tokens = cfg.get("max_tokens", 4096)
        timeout = cfg.get("timeout", 120)

        if provider == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=model,
                api_key=cfg.get("api_key"),
                base_url=cfg.get("base_url"),
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                max_retries=3,
            )

        if provider in ("openai_compatible", "doubao"):
            # 豆包 / DeepSeek / Qwen / Ollama 等 OpenAI 兼容接口统一用 ChatOpenAI
            from langchain_openai import ChatOpenAI

            default_base_url = (
                "https://ark.cn-beijing.volces.com/api/v3" if provider == "doubao" else None
            )
            return ChatOpenAI(
                model=model,
                api_key=cfg.get("api_key") or "placeholder",
                base_url=cfg.get("base_url", default_base_url),
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                max_retries=3,
            )

        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(
                model=model,
                api_key=cfg.get("api_key"),
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                max_retries=3,
            )

        if provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=model,
                google_api_key=cfg.get("api_key"),
                temperature=temperature,
                max_output_tokens=max_tokens,
                timeout=timeout,
            )

        raise UnknownProviderError(f"不支持的 provider: {provider}")

    # ── 配置加载与环境变量展开 ──
    @classmethod
    def _load_config_cached(cls, path: str) -> Dict[str, Any]:
        config_path = Path(path).resolve()
        now = time.time()

        if path in cls._config_cache:
            cached, mtime, ts = cls._config_cache[path]
            if config_path.exists() and config_path.stat().st_mtime <= mtime:
                if now - ts < cls._config_cache_ttl:
                    return cached

        config = cls._load_config(path)
        if config_path.exists():
            cls._config_cache[path] = (config, config_path.stat().st_mtime, now)
        return config

    @staticmethod
    def _load_config(path: str) -> Dict[str, Any]:
        config_path = Path(path).resolve()
        if not config_path.exists():
            raise ConfigLoadError(f"配置文件不存在: {path}")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigLoadError(f"YAML 解析错误: {e}") from e

        if not isinstance(config, dict):
            raise ConfigLoadError(f"配置文件格式无效: {path}")
        return ModelRouter._expand_env_vars(config)

    @staticmethod
    def _expand_env_vars(obj: Any) -> Any:
        """递归展开 `${VAR}` 与 `${VAR:default}` 占位符。"""
        if isinstance(obj, dict):
            return {k: ModelRouter._expand_env_vars(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [ModelRouter._expand_env_vars(v) for v in obj]
        if isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            expr = obj[2:-1]
            if ":" in expr:
                name, default = expr.split(":", 1)
                return os.getenv(name.strip(), default)
            name = expr.strip()
            value = os.getenv(name)
            if value is None:
                logger.warning(f"环境变量未设置: {name}")
            return value
        return obj
