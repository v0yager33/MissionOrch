"""基于 pydantic-settings 的统一配置入口。

优先级（从高到低）：
  1. 环境变量（前缀 MISSIONORCH_）
  2. `.env` 文件
  3. 代码内默认值

外部的 YAML 配置（models.yaml / agents.yaml / rag.yaml）仍由各模块自行加载，
这里只管理**应用级配置**：工作流参数、日志、追踪、可观测性等。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class WorkflowSettings(BaseSettings):
    """迭代循环相关参数。"""

    max_iterations: int = Field(default=3, ge=1, le=20, description="最大迭代次数")
    quality_threshold: float = Field(default=8.0, ge=0.0, le=10.0, description="质量阈值，达到后停止")
    early_stop: bool = Field(default=True, description="是否在达到阈值后立即停止")

    model_config = SettingsConfigDict(env_prefix="MISSIONORCH_WORKFLOW_", extra="ignore")


class TracingSettings(BaseSettings):
    """LangSmith / 追踪相关配置。

    当 `langsmith_api_key` 设置时，会自动开启 LangChain tracing v2。
    """

    langsmith_api_key: Optional[str] = Field(default=None, description="LangSmith API Key")
    langsmith_project: str = Field(
        default="missionorch-lc", description="LangSmith 项目名"
    )
    tracing_enabled: bool = Field(
        default=False, description="是否启用 LangChain tracing（自动根据 API key 判定）"
    )

    model_config = SettingsConfigDict(env_prefix="MISSIONORCH_TRACING_", extra="ignore")

    def activate(self) -> None:
        """把配置写入 LangChain 约定的环境变量。"""
        import os

        if self.langsmith_api_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = self.langsmith_api_key
            os.environ["LANGCHAIN_PROJECT"] = self.langsmith_project
            self.tracing_enabled = True


class AppSettings(BaseSettings):
    """应用级配置总入口。"""

    # 路径相关
    config_dir: Path = Field(default=Path("config"), description="YAML 配置目录")
    log_dir: Path = Field(default=Path("logs"), description="日志目录")

    # YAML 文件路径
    agents_config: Path = Field(default=Path("config/agents.yaml"))
    models_config: Path = Field(default=Path("config/models.yaml"))
    rag_config: Path = Field(default=Path("config/rag.yaml"))

    # LLM 容错
    llm_max_retries: int = Field(default=3, ge=0, le=10)
    llm_request_timeout: int = Field(default=120, ge=10)

    workflow: WorkflowSettings = Field(default_factory=WorkflowSettings)
    tracing: TracingSettings = Field(default_factory=TracingSettings)

    model_config = SettingsConfigDict(
        env_prefix="MISSIONORCH_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# 单例
_settings: Optional[AppSettings] = None


def get_settings() -> AppSettings:
    """获取全局配置单例（首次调用时初始化并激活 tracing）。"""
    global _settings
    if _settings is None:
        _settings = AppSettings()
        _settings.tracing.activate()
    return _settings


def reset_settings() -> None:
    """主要用于测试：重置单例。"""
    global _settings
    _settings = None
