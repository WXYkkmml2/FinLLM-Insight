# FinLLM-Insight

<p align="center">
    <strong>基于大语言模型的美国上市公司财报分析与股票预测系统</strong>
</p>

<p align="center">
    <a href="#项目概述">项目概述</a> •
    <a href="#系统架构">系统架构</a> •
    <a href="#核心功能">核心功能</a> •
    <a href="#快速开始">快速开始</a> •
    <a href="#运行指南">运行指南</a> •
    <a href="#技术实现">技术实现</a> •
    <a href="#项目结构">项目结构</a> •
    <a href="#贡献指南">贡献指南</a> •
    <a href="#鸣谢">鸣谢</a> •
    <a href="#许可证">许可证</a>
</p>

## 项目概述

FinLLM-Insight 是一个创新的投资辅助系统，利用大语言模型（LLM）对美国上市公司年度报告（10-K文件）进行深度分析，并结合机器学习技术进行股票收益预测。系统通过检索增强生成（RAG）技术，提供交互式的年报查询功能，帮助投资者更高效地分析企业基本面，作出更明智的投资决策。

### 项目背景

传统的财报分析耗时费力，投资者难以快速有效地从美国上市公司的10-K报告中提取关键信息。这些报告通常篇幅冗长，包含大量细节内容。大型语言模型的出现为自动化分析复杂财务文档提供了可能性，但如何将其能力转化为实际的投资洞察仍然具有挑战性。本项目旨在搭建一个完整的系统，将大语言模型的自然语言理解能力与量化投资方法结合，创造一个智能投资辅助工具。

## 系统架构

FinLLM-Insight 采用模块化设计，由以下主要部分组成：

1. **数据获取模块**：自动从SEC和Financial Modeling Prep API下载美国上市公司10-K报告
2. **文本处理模块**：将HTML年报转换为结构化文本并进行预处理
3. **目标变量生成**：基于历史股价数据生成预测目标
4. **嵌入生成模块**：将财报文本转换为向量表示
5. **LLM特征分析**：使用大语言模型对年报内容进行多维度分析
6. **模型训练模块**：基于LLM生成的特征训练预测模型
7. **预测与推荐**：生成股票收益预测和投资建议
8. **检索增强生成**：实现与年报内容的自然语言交互

## 核心功能

### 1. 自动化数据获取

- 支持S&P500指数成分股（可扩展至S&P400、S&P600）
- 自动下载最新和历史10-K报告
- 支持增量更新数据

### 2. 智能财报分析

- 利用大语言模型深度分析10-K报告内容
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

### 环境要求

- Python 3.8+
- OpenAI API密钥
- Financial Modeling Prep API密钥
- 足够的存储空间用于年报数据（建议至少5GB）

### 基础配置

1. 克隆仓库并安装依赖：

```bash
git clone https://github.com/WXYkkmml2/FinLLM-Insight.git
cd FinLLM-Insight
pip install -r requirements.txt
```

2. 配置API密钥：

创建一个`.env`文件并添加你的API密钥：

```
OPENAI_API_KEY=your_openai_api_key_here
```

3. 在`config/config.json`中配置Financial Modeling Prep API密钥：

```json
{
  "financial_modelling_prep_api_key": "your_fmp_api_key_here",
  ...
}
```

## 运行指南

### 本地环境运行

#### 环境准备

```bash
# 克隆仓库
git clone https://github.com/WXYkkmml2/FinLLM-Insight.git
cd FinLLM-Insight

# 创建虚拟环境 (推荐使用conda)
conda create -n finllm python=3.9
conda activate finllm

# 安装依赖
pip install -r requirements.txt

# 配置API密钥（打开.env文件并添加你的OpenAI密钥）
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env

# 编辑config/config.json，添加Financial Modeling Prep API密钥
# 将"financial_modelling_prep_api_key"字段更新为你的密钥
```

#### 分步运行

按照数据处理流水线顺序依次执行以下命令：

```bash
# 步骤1: 下载美国上市公司10-K报告
python src/data/download_reports.py --max_stocks 10  # 从S&P500中选取10家公司测试

# 步骤2: 转换报告格式 (HTML到文本)
python src/data/convert_formats.py

# 步骤3: 生成目标变量 (基于股票价格)
python src/data/make_targets.py

# 步骤4: 生成文本嵌入向量
export HUGGINGFACE_API_KEY='？？？'
python src/features/embeddings.py

# 步骤5: 使用LLM生成结构化特征
python src/features/llm_features.py

# 步骤6: 训练预测模型
python src/models/train.py --model_type regression --model_name random_forest

# 步骤7: 生成预测结果
python src/models/predict.py
```

#### 完整流水线一次性运行

```bash
# 运行完整流水线
python src/pipeline.py

# 或者跳过特定步骤（例如跳过步骤1和2）
python src/pipeline.py --skip_steps "1,2"

# 或者只运行特定步骤
python src/pipeline.py --step 5  # 只运行步骤5（生成LLM特征）
```

#### 交互式年报查询

完成数据处理和模型训练后，可以使用交互式查询功能：

```bash
# 启动交互式查询
python src/rag/rag_component.py --interactive
```

### 在Kaggle/Colab上运行

> **⚠️ 重要提示：** 由于SEC和Financial Modeling Prep网站的访问限制，在Kaggle、Colab等云服务器环境中直接运行`download_reports.py`步骤可能会遇到403错误（禁止访问）。建议在本地环境下载年报数据，然后将数据上传到云环境，或直接从步骤2开始运行。



### 注意事项

1. **API密钥限制**：OpenAI和Financial Modeling Prep API都有使用限制，大量处理可能产生费用。建议先使用小型数据集测试。

2. **处理时间**：完整流水线包含多个耗时步骤，特别是LLM特征生成。在云环境中可能需要注意运行时间限制。

3. **存储空间**：下载的年报和生成的向量数据可能占用较大空间。在本地可以降低`max_stocks`参数；在云环境中需要关注存储限制。

4. **GPU加速**：`embeddings.py`和模型训练步骤可以从GPU加速中获益。如果可用，推荐在有GPU的环境中运行。

5. **断点续传**：如果处理中断，可以使用`--skip_steps`参数跳过已完成的步骤，从中断处继续处理。

6. **调试模式**：添加`--debug`参数可以开启详细日志，有助于问题排查。

7. **SEC访问限制**：由于SEC官方网站对云服务器IP有访问限制，在Kaggle/Colab等环境中直接下载年报可能会遇到403错误。建议在本地下载数据后上传到云环境，或使用预处理好的数据。

```bash
python src/pipeline.py --debug
```

## 技术实现

项目使用了以下主要技术：

- **数据获取**: Python requests, BeautifulSoup4
- **文本处理**: Python re, NLTK
- **向量嵌入**: Sentence-Transformers
- **向量数据库**: ChromaDB
- **机器学习**: Scikit-learn, XGBoost, LightGBM
- **大语言模型**: OpenAI API
- **检索增强生成**: LangChain
- **数据可视化**: Matplotlib, Seaborn

## 项目结构

```
FinLLM-Insight/
├── config/                # 配置文件
│   ├── config.json        # 全局配置
│   └── questions.json     # LLM分析问题
├── data/                  # 数据存储
│   ├── raw/               # 原始年报数据
│   └── processed/         # 处理后的数据
├── models/                # 保存训练好的模型
├── notebooks/             # Jupyter笔记本
├── predictions/           # 预测结果输出
├── src/                   # 源代码
│   ├── data/              # 数据处理模块
│   ├── features/          # 特征工程模块
│   ├── models/            # 模型训练和预测
│   ├── rag/               # 检索增强生成模块
│   └── pipeline.py        # 完整处理流水线
├── .env                   # 环境变量
├── LICENSE                # 许可证
├── README.md              # 项目说明
├── requirements.txt       # 依赖项
└── setup.py               # 安装脚本
```


## 鸣谢

- [Financial Modeling Prep API](https://financialmodelingprep.com/developer/docs/) - 提供财务数据API
- [U.S. Securities and Exchange Commission](https://www.sec.gov/) - 提供10-K报告
- [OpenAI](https://openai.com/) - 提供大语言模型API
- [Hugging Face](https://huggingface.co/) - 提供开源NLP模型
- [Sentence-Transformers](https://www.sbert.net/) - 提供文本嵌入模型
- [ChromaDB](https://www.trychroma.com/) - 提供向量数据库

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件
