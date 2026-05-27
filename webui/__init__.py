"""MissionOrch-LC Streamlit Web UI 包。

模块组织：
    app.py                          —— 主入口（多页面 router）
    state.py                        —— Streamlit session_state 包装
    components/                     —— 复用组件
        config_io.py                —— YAML / .env 读写
        runner.py                   —— 后端任务运行（asyncio + 阶段进度回调）
        rag_storage_inspector.py    —— RAG 索引产物统计
        rag_query.py                —— RAG 直查封装
        env_io.py                   —— .env 文件读写（API Keys）
        ui_helpers.py               —— Streamlit 公共渲染辅助
    pages/                          —— 子页面
        1_🚀_任务执行.py
        2_📊_结果详情.py
        3_🔍_RAG_直查.py
        4_⚙️_RAG_配置.py
        5_🔑_LLM_API_配置.py
        6_📚_知识库管理.py
        7_📜_历史运行.py
"""

from __future__ import annotations
