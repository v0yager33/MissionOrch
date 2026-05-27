#!/usr/bin/env python3
"""一键启动 MissionOrch-LC Web UI。

用法：
    python run_ui.py                     # 默认 http://localhost:8501
    python run_ui.py --port 8888         # 指定端口
    python run_ui.py --host 0.0.0.0      # 对外开放
    python run_ui.py --no-browser        # 不自动打开浏览器

行为：
- 自动把项目根 + src/ 注入 PYTHONPATH
- 自动从项目根 ``.env`` 读取 API key 注入进程
- 用 ``streamlit.web.cli`` 直接拉起 ``webui/app.py``
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP_PATH = ROOT / "webui" / "app.py"


def _inject_paths() -> None:
    src_dir = ROOT / "src"
    paths = [str(ROOT), str(src_dir)]
    existing = os.environ.get("PYTHONPATH", "")
    new_pythonpath = os.pathsep.join([p for p in paths if p] + ([existing] if existing else []))
    os.environ["PYTHONPATH"] = new_pythonpath

    for path in paths:
        if path not in sys.path:
            sys.path.insert(0, path)


def _load_dotenv() -> None:
    """简单解析项目根 .env 注入到当前进程。"""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


def main() -> int:
    parser = argparse.ArgumentParser(description="启动 MissionOrch-LC Web UI")
    parser.add_argument("--host", default="localhost", help="绑定地址，默认 localhost")
    parser.add_argument("--port", type=int, default=8501, help="端口，默认 8501")
    parser.add_argument(
        "--no-browser", action="store_true", help="启动后不自动打开浏览器"
    )
    args = parser.parse_args()

    if not APP_PATH.exists():
        print(f"❌ 找不到 Web UI 入口：{APP_PATH}", file=sys.stderr)
        return 1

    _inject_paths()
    _load_dotenv()

    # 通过 streamlit.web.cli 直接拉起，避免依赖 shell 命令
    streamlit_args = [
        "streamlit",
        "run",
        str(APP_PATH),
        "--server.address",
        args.host,
        "--server.port",
        str(args.port),
        "--server.headless",
        "true" if args.no_browser else "false",
        "--browser.gatherUsageStats",
        "false",
    ]
    sys.argv = streamlit_args
    print(f"🚀 启动 Streamlit： http://{args.host}:{args.port}")
    print(f"   入口文件: {APP_PATH}")
    print(f"   PYTHONPATH: {os.environ.get('PYTHONPATH', '')}")

    from streamlit.web import cli as stcli

    return stcli.main()  # type: ignore[no-any-return]


if __name__ == "__main__":
    sys.exit(main() or 0)
