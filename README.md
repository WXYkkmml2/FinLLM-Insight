# FinLLM-Insight


<p align="center">
    <strong>基于大语言模型的财报分析与股票预测系统</strong>
</p>

<p align="center">
    <a href="#项目概述">项目概述</a> •
    <a href="#系统架构">系统架构</a> •
    <a href="#核心功能">核心功能</a> •
    <a href="#快速开始">快速开始</a> •
    <a href="#使用指南">使用指南</a> •
    <a href="#技术实现">技术实现</a> •
    <a href="#项目结构">项目结构</a> •
    <a href="#贡献指南">贡献指南</a> •
    <a href="#鸣谢">鸣谢</a> •
    <a href="#许可证">许可证</a>
</p>

## 项目概述

FinLLM-Insight 是一个创新的投资辅助系统，利用大语言模型（LLM）对上市公司年度报告进行深度分析，并结合机器学习技术进行股票收益预测。系统通过检索增强生成（RAG）技术，提供交互式的年报查询功能，帮助投资者更高效地分析企业基本面，作出更明智的投资决策。

### 项目背景

传统的财报分析耗时费力，投资者难以快速有效地从大量文本中提取关键信息。大型语言模型的出现为自动化分析复杂财务文档提供了可能性，但如何将其能力转化为实际的投资洞察仍然具有挑战性。本项目旨在搭建一个完整的系统，将大语言模型的自然语言理解能力与量化投资方法结合，创造一个智能投资辅助工具。

## 系统架构


FinLLM-Insight 采用模块化设计，由以下主要部分组成：

1. **数据获取模块**：自动从巨潮资讯网下载中国上市公司年度报告
2. **文本处理模块**：将PDF年报转换为结构化文本并进行预处理
3. **目标变量生成**：基于历史股价数据生成预测目标
4. **嵌入生成模块**：将财报文本转换为向量表示
5. **LLM特征分析**：使用大语言模型对年报内容进行多维度分析
6. **模型训练模块**：基于LLM生成的特征训练预测模型
7. **预测与推荐**：生成股票收益预测和投资建议
8. **检索增强生成**：实现与年报内容的自然语言交互

## 核心功能

### 1. 自动化数据获取

- 支持沪深300指数成分股
- 自动下载最新和历史年报
- 支持增量更新数据

### 2. 智能财报分析

- 利用大语言模型深度分析财报内容
- 提供多维度企业分析（财务健康、商业模式、增长前景等）
- 生成结构化评分和投资建议

### 3. 收益率预测

- 基于LLM特征训练机器学习模型
- 支持不同时间窗口的收益预测
- 提供预测可视化和解释

### 4. 交互式年报查询

- 基于RAG技术的自然语言查询
- 快速定位关键信息
- 提供信息来源的溯源

## 快速开始

### 系统要求

- Python 3.8+
- 8GB+ RAM
- 互联网连接

### 安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/FinLLM-Insight.git
cd FinLLM-Insight

# 安装依赖
pip install -r requirements.txt
```

### 环境配置

1. 创建环境变量文件：
```bash
cp .env.example .env
```

2. 编辑 `.env` 文件，添加你的 API 密钥：
```
OPENAI_API_KEY=your_openai_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
```

3. 根据需要调整 `config/config.json` 配置

### 运行

```bash
# 运行完整处理流水线
python src/pipeline.py

# 运行交互式查询系统
python src/rag/rag_component.py --interactive
```

## 使用指南

### 数据获取

```bash
# 下载沪深300成分股年报
python src/data/download_reports.py
```

### 特征生成

```bash
# 生成文本嵌入
python src/features/embeddings.py

# 生成LLM特征
python src/features/llm_features.py
```

### 模型训练

```bash
# 训练回归模型
python src/models/train.py --model_type regression --model_name random_forest

# 训练分类模型
python src/models/train.py --model_type classification --model_name random_forest
```

### 预测

```bash
# 使用最新模型进行预测
python src/models/predict.py
```

### RAG查询

```bash
# 启动交互式查询
python src/rag/rag_component.py --interactive
```

## 技术实现

- **数据源**：AKShare API连接巨潮资讯网
- **文本处理**：PyPDF2, pdfplumber, HanziConv
- **向量数据库**：ChromaDB
- **嵌入模型**：BAAI/bge-large-zh-v1.5
- **大语言模型**：GPT-3.5-Turbo
- **机器学习**：scikit-learn, RandomForest, XGBoost
- **可视化**：Matplotlib, Seaborn

## 项目结构

```
FinLLM-Insight/
├── config/                 # 配置文件
│   ├── config.json         # 主配置文件
│   └── questions.json      # LLM分析问题
├── data/                   # 数据目录
│   ├── raw/                # 原始数据
│   └── processed/          # 处理后数据
├── models/                 # 保存训练好的模型
├── predictions/            # 预测结果目录
├── src/                    # 源代码
│   ├── data/               # 数据获取与处理
│   ├── features/           # 特征工程
│   ├── models/             # 模型训练与预测
│   ├── rag/                # 检索增强生成
│   └── pipeline.py         # 完整处理流水线
├── notebooks/              # Jupyter笔记本
├── tests/                  # 测试代码
├── docs/                   # 文档
├── .env.example            # 环境变量示例文件
├── LICENSE                 # 许可证
├── README.md               # 项目说明
├── requirements.txt        # 依赖列表
└── setup.py                # 安装脚本
```

## 常见问题

### API密钥配置

如果遇到"API key not found"或类似错误，请检查：
1. 确认已创建并正确配置 `.env` 文件
2. 确认安装了 `python-dotenv`（包含在 requirements.txt 中）
3. 重新启动应用程序以加载环境变量

### 数据下载问题

如果年报下载失败：
1. 检查网络连接
2. 确认巨潮资讯网API的可用性
3. 尝试增加 `download_delay` 参数的值以减少请求频率

## 贡献指南

欢迎贡献代码、报告问题或提出改进建议！请参阅 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详情。

## 鸣谢

- 本项目受到 [GPT-InvestAR](https://github.com/UditGupta10/GPT-InvestAR) 的启发
- 感谢CSC6052/5051/4100/DDA6307/MDS5110 NLP课程的支持
- 使用了 [AKShare](https://github.com/akfamily/akshare) 提供的财经数据接口

## 许可证

[MIT](LICENSE) © WXYkkmml2




