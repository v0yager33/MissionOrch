# MissionOrch - 多智能体任务编排系统

MissionOrch 是一个多智能体任务编排系统，专门用于生成战役级行动方案（COA - Course of Action）。系统采用规划-评估-反思-验证的迭代循环来不断改进生成的行动方案。

## 系统架构

项目由以下几个核心组件构成：

1. **智能体系统**：
   - 规划智能体（Planner Agent）：负责生成初始COA
   - 评估智能体（Judge Agent）：评估COA质量并给出分数
   - 反思智能体（Reflector Agent）：分析缺陷并提供改进建议
   - 验证智能体（Validator Agent）：验证COA格式合理性并提取纯COA矩阵

2. **模型路由系统**：支持多种AI模型提供商（OpenAI、Gemini、Anthropic、豆包等）

3. **RAG系统**：知识检索增强，支持文档、图像等多种格式

4. **编排器**：协调四个智能体的工作流程

## 安装依赖

```bash
pip install -r requirements.txt
```

## 配置API密钥

在运行系统之前，需要设置相应的API密钥环境变量：

```bash
# OpenAI API密钥
export OPENAI_API_KEY="your_openai_api_key"

# Google Gemini API密钥
export GEMINI_API_KEY="your_gemini_api_key"

# Anthropic Claude API密钥
export ANTHROPIC_API_KEY="your_anthropic_api_key"

# 字节跳动豆包（Doubao）API密钥
export ARK_API_KEY="your_ark_api_key"

# 阿里云通义千问API密钥
export DASHSCOPE_API_KEY="your_dashscope_api_key"

# DeepSeek API密钥
export DEEPSEEK_API_KEY="your_deepseek_api_key"
```

## 运行测试

在设置好API密钥后，可以运行测试来验证系统：

```bash
python test_full_system.py
```

或者运行示例：

```bash
python example_usage.py
```

## 使用系统

您可以使用以下方式运行系统：

```bash
python main.py
```

## 项目结构

```
MissionOrch/
├── src/
│   └── missionorch/
│       ├── core/           # 核心模块
│       ├── schemas/        # 数据模型定义
│       ├── prompts/        # 提示词模板
│       └── utils/          # 工具函数
├── config/                 # 配置文件
├── prompts/                # 提示词文件（外部链接）
├── pyproject.toml          # 项目配置
├── requirements.txt        # 依赖列表
└── README.md               # 项目说明
```

## 功能特点

1. **人类可读COA**：生成结构化的战役级行动方案
2. **仿真矩阵**：输出适合仿真系统使用的矩阵格式
3. **多格式输出**：支持JSON、YAML、矩阵表格等多种格式
4. **事件驱动**：基于事件和条件的动态矩阵系统
5. **迭代优化**：通过评估-反思循环持续改进COA质量
6. **智能验证**：自动验证COA格式合理性并提取纯矩阵数据