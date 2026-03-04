"""
日志配置模块

每次调用 setup_logging() 会创建带时间戳的独立日志文件：
  logs/debug_20260101_232131_314241.log    — 全量日志（含 Agent 输入输出）
  logs/interactions_20260101_232131_314241.log — 仅 Agent 输入输出
"""

import logging
from datetime import datetime
from pathlib import Path


def setup_logging(log_dir: str = "logs", console_level: int = logging.INFO) -> str:
    """
    配置日志系统，每次调用生成独立的带时间戳日志文件。

    Args:
        log_dir: 日志目录路径
        console_level: 控制台日志级别

    Returns:
        本次日志文件的时间戳标识（如 20260101_232131_314241）
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    debug_log_file = log_path / f"debug_{timestamp}.log"
    interaction_log_file = log_path / f"interactions_{timestamp}.log"

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    interaction_format = "%(asctime)s - %(message)s"

    # 清除 root logger 上已有的 handler，避免重复
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 控制台 handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(console_handler)

    # 全量 debug 日志文件 handler
    debug_file_handler = logging.FileHandler(str(debug_log_file), mode="w", encoding="utf-8")
    debug_file_handler.setLevel(logging.DEBUG)
    debug_file_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(debug_file_handler)

    # 交互日志（Agent 输入输出）独立文件 handler
    interaction_logger = logging.getLogger("missionorch.interactions")
    # 清除已有 handler
    for handler in interaction_logger.handlers[:]:
        interaction_logger.removeHandler(handler)
    interaction_logger.setLevel(logging.DEBUG)
    interaction_logger.propagate = True  # 同时写入全量日志

    interaction_file_handler = logging.FileHandler(str(interaction_log_file), mode="w", encoding="utf-8")
    interaction_file_handler.setLevel(logging.DEBUG)
    interaction_file_handler.setFormatter(logging.Formatter(interaction_format))
    interaction_logger.addHandler(interaction_file_handler)

    logging.getLogger(__name__).info(
        f"Logging initialized — debug: {debug_log_file}, interactions: {interaction_log_file}"
    )

    return timestamp
