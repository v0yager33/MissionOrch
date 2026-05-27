"""LLM API 配置页 —— 编辑 config/models.yaml + .env 中的 API keys + ping 测试。"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict

import streamlit as st

from webui.components.config_io import load_yaml, save_yaml
from webui.components.env_io import (
    KNOWN_API_KEYS,
    load_env,
    merge_env_into_process,
    save_env,
)
from webui.components.ui_helpers import render_header
from webui.state import init_state

init_state()
render_header(
    "🔑 LLM API 配置",
    "编辑各 Provider 的连接参数（model / api_key / base_url）+ 测试连通性。",
)

MODELS_YAML = "config/models.yaml"
ENV_PATH = Path(".env")


# ── 内部：ping 单个 model_id ──
def _ping_model(model_id: str) -> tuple[float, bool, str]:
    """对指定 model_id 发一条最小请求，验证连通性。

    Returns:
        (耗时秒, 是否成功, 信息片段)
    """
    started = time.time()
    try:
        from langchain_core.messages import HumanMessage

        from missionorch_lc.core.model_router import ModelRouter

        # 清缓存确保读最新 yaml + .env
        ModelRouter.clear_cache()
        chat = ModelRouter.get(model_id)
        response = chat.invoke([HumanMessage(content="ping")])
        text = (getattr(response, "content", "") or "").strip()
        snippet = text[:80] if text else "(空响应)"
        return time.time() - started, True, snippet
    except Exception as ping_error:
        return time.time() - started, False, f"{type(ping_error).__name__}: {ping_error}"


# ── Section 1: API Keys（.env） ──
st.markdown("### 🗝️ API Keys（写入项目根目录 `.env`）")
existing_env = load_env(ENV_PATH)

key_updates: Dict[str, str] = {}
env_cols = st.columns(2)
for index, key in enumerate(KNOWN_API_KEYS):
    col = env_cols[index % 2]
    current = existing_env.get(key) or os.getenv(key) or ""
    masked = current[:6] + "…" + current[-4:] if len(current) > 12 else current
    new_val = col.text_input(
        key,
        value=current,
        type="password" if "KEY" in key else "default",
        help=f"环境变量 ${{ {key} }}。当前值（脱敏）：{masked or '（未设置）'}",
        key=f"env_{key}",
    )
    key_updates[key] = new_val

env_save_col1, env_save_col2 = st.columns([1, 5])
with env_save_col1:
    if st.button("💾 保存到 .env", type="primary"):
        try:
            cleaned = {k: v for k, v in key_updates.items() if v is not None}
            save_env(ENV_PATH, cleaned)
            merge_env_into_process(cleaned)
            st.success(f"✅ 已写入 {ENV_PATH}，并已注入当前进程。")
        except Exception as save_error:
            st.error(f"❌ 保存失败：{save_error}")
with env_save_col2:
    st.caption(
        "**注意**：写入 `.env` 后会立即注入当前 Streamlit 进程，无需重启即可让 ModelRouter 读到新 key。"
    )

st.divider()


# ── Section 2: models.yaml ──
st.markdown("### 🧩 LLM Provider 列表（`config/models.yaml`）")

models_raw = load_yaml(MODELS_YAML) or {}
models_cfg: Dict[str, Any] = dict(models_raw.get("models") or {})

if not models_cfg:
    st.warning("`config/models.yaml` 中没有任何 model 定义。")

new_models: Dict[str, Any] = {}
for model_id, model_def in models_cfg.items():
    with st.expander(f"🧠 `{model_id}` — provider={model_def.get('provider', '?')}", expanded=False):
        keep = st.checkbox("保留", value=True, key=f"keep_model_{model_id}")
        if not keep:
            continue

        cols = st.columns(2)
        provider = cols[0].selectbox(
            "provider",
            ["openai", "openai_compatible", "doubao", "anthropic", "gemini"],
            index=["openai", "openai_compatible", "doubao", "anthropic", "gemini"].index(
                str(model_def.get("provider", "openai_compatible"))
            ),
            key=f"provider_{model_id}",
        )
        model_name = cols[1].text_input(
            "model（厂商侧模型名）",
            value=str(model_def.get("model", "")),
            key=f"model_{model_id}",
        )

        api_key = st.text_input(
            "api_key（支持 `${ENV_VAR:default}` 占位）",
            value=str(model_def.get("api_key") or ""),
            key=f"key_{model_id}",
            help="**强烈建议**写成 `${OPENAI_API_KEY}` 这种形式，把真实 key 放进 `.env`。",
        )
        base_url = st.text_input(
            "base_url（可选）",
            value=str(model_def.get("base_url") or ""),
            key=f"baseurl_{model_id}",
        )

        param_cols = st.columns(4)
        temperature = param_cols[0].slider(
            "default_temperature", 0.0, 1.5,
            float(model_def.get("default_temperature", 0.3)), step=0.05,
            key=f"temp_{model_id}",
        )
        max_tokens = param_cols[1].number_input(
            "max_tokens", 256, 32768,
            int(model_def.get("max_tokens", 4096)), step=256,
            key=f"maxtok_{model_id}",
        )
        timeout = param_cols[2].number_input(
            "timeout(秒)", 10, 600,
            int(model_def.get("timeout", 120)), step=10,
            key=f"timeout_{model_id}",
        )
        max_retries = param_cols[3].number_input(
            "max_retries", 0, 10,
            int(model_def.get("max_retries", 3)),
            key=f"retries_{model_id}",
        )

        # 保留 extra_body / reasoning_effort / fallback_model_ids
        extra_body = model_def.get("extra_body")
        reasoning_effort = model_def.get("reasoning_effort")
        fallback_ids = model_def.get("fallback_model_ids")

        new_def: Dict[str, Any] = {
            "provider": provider,
            "model": model_name,
            "api_key": api_key,
            "base_url": base_url or None,
            "default_temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout,
            "max_retries": max_retries,
        }
        if not new_def["base_url"]:
            new_def.pop("base_url")
        if extra_body:
            new_def["extra_body"] = extra_body
        if reasoning_effort:
            new_def["reasoning_effort"] = reasoning_effort
        if fallback_ids:
            new_def["fallback_model_ids"] = fallback_ids

        new_models[model_id] = new_def

        # ── Ping 测试 ──
        ping_col1, ping_col2 = st.columns([1, 5])
        with ping_col1:
            if st.button(f"📡 Ping {model_id}", key=f"ping_{model_id}"):
                with st.spinner("正在 ping…"):
                    elapsed, ok, message = _ping_model(model_id)
                if ok:
                    st.success(f"✅ {model_id} 可用（{elapsed:.2f}s）：{message}")
                else:
                    st.error(f"❌ {model_id} 不可用（{elapsed:.2f}s）：{message}")
        with ping_col2:
            st.caption(
                "Ping 会用当前 `.env` + `models.yaml` 实际发一条 1 token 的请求验证连通性。"
            )

# 保存
st.divider()
save_col1, save_col2 = st.columns([1, 5])
with save_col1:
    if st.button("💾 保存 models.yaml", type="primary", key="save_models_yaml"):
        try:
            models_raw["models"] = new_models
            save_yaml(MODELS_YAML, models_raw)
            # 清缓存
            try:
                from missionorch_lc.core.model_router import ModelRouter
                ModelRouter.clear_cache()
            except Exception:
                pass
            st.success(f"✅ 已写入 {MODELS_YAML}，ModelRouter 缓存已清空。")
        except Exception as save_error:
            st.error(f"❌ 保存失败：{save_error}")
with save_col2:
    st.caption("保存后已自动清空 ModelRouter 单例缓存，下一次调用就会用新配置。")
