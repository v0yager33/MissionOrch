#!/usr/bin/env bash
# Web UI 冒烟测试：起 streamlit → curl 验证主页 → kill 进程
# 用法： bash scripts/smoke_webui.sh [PORT]

set -uo pipefail

PORT="${1:-10500}"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_FILE="/tmp/streamlit_smoke_${PORT}.log"
HOME_HTML="/tmp/streamlit_smoke_home_${PORT}.html"

cd "$ROOT_DIR"

# 1. 起进程（脱离 shell session）
echo "[smoke] starting streamlit on 127.0.0.1:${PORT} ..."
setsid bash -c "source .venv/bin/activate && python run_ui.py --host 127.0.0.1 --port ${PORT} --no-browser" \
    > "$LOG_FILE" 2>&1 < /dev/null &
PID=$!
disown $PID 2>/dev/null || true

# 2. 等启动
for i in $(seq 1 15); do
    sleep 1
    if grep -q "URL: http" "$LOG_FILE" 2>/dev/null; then
        echo "[smoke] streamlit ready after ${i}s"
        break
    fi
done

# 3. curl 主页
echo "[smoke] --- curl home ---"
curl -s -o "$HOME_HTML" -w 'HTTP=%{http_code}  size=%{size_download}\n' --max-time 10 \
    "http://127.0.0.1:${PORT}/" || echo "[smoke] curl failed"

echo "[smoke] --- HTML head (5 lines) ---"
head -n 5 "$HOME_HTML" 2>/dev/null | cat || true

# 4. 检查 _stcore/health（streamlit 内置健康检查端点）
echo "[smoke] --- health endpoint ---"
curl -s -w '\nHTTP=%{http_code}\n' --max-time 5 \
    "http://127.0.0.1:${PORT}/_stcore/health" || echo "[smoke] health failed"

# 5. 关掉
echo "[smoke] --- shutdown ---"
pkill -P $PID 2>/dev/null || true
kill $PID 2>/dev/null || true
sleep 1
pkill -9 -f "run_ui.py --host 127.0.0.1 --port ${PORT}" 2>/dev/null || true

# 6. 末尾日志
echo "[smoke] --- streamlit log tail ---"
tail -n 15 "$LOG_FILE" | cat

echo "[smoke] done"
