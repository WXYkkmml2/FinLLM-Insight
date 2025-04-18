# FinLLM-Insight
基于大语言模型的财报分析与股票预测系统

## 项目简介
FinLLM-Insight是一个利用大型语言模型(LLM)分析公司年度报告并预测股票投资价值的系统。该项目整合了自然语言处理、检索增强生成(RAG)和机器学习技术，为投资者提供数据驱动的决策支持。
## 主要特点
**自动数据获取**：从中国证券市场自动下载上市公司年度报告
**智能文本分析**：利用大型语言模型深度分析财务报告内容
**特征工程**：基于LLM分析生成结构化特征
**投资预测**：预测股票未来表现，提供投资建议
**交互式查询**：实现基于RAG的智能问答系统，支持对财报内容的交互式探索

## 快速开始
### 环境要求

- Python 3.8+
- 依赖包: 见`requirements.txt`
- 
### 安装
```bash
# 克隆仓库
git clone https://github.com/yourusername/FinLLM-Insight.git
cd FinLLM-Insight

# 安装依赖
pip install -r requirements.txt
```

### 配置

在`config/config.json`中配置您的参数:

```json
{
    "annual_reports_html_save_directory": "./data/raw/annual_reports",
    "china_stock_index": "沪深300",
    "min_year": 2018,
    "download_delay": 2
}
```

### 使用方法

#### 方法一: 运行各模块

```bash
# 下载年报
python src/data/download_reports.py --config_path config/config.json

# 生成嵌入向量
python src/features/embeddings.py --config_path config/config.json

# 更多步骤...
```

#### 方法二: Jupyter Notebook

打开`notebooks/`目录下的Jupyter notebooks，跟随其中的说明执行完整流程。

## 项目模块说明

### 数据获取与预处理

使用AKShare API获取中国上市公司的财务报告，并进行格式转换和预处理。

### 文本嵌入与特征生成

将财务报告文本转换为向量表示，并利用大语言模型生成结构化特征。

### 模型训练与评估

基于生成的特征训练预测模型，评估模型性能并生成投资建议。

### RAG问答系统

实现检索增强生成(RAG)系统，支持对财报内容的自然语言查询。

## 实验结果

此处简要展示系统性能和分析结果...

## 贡献指南

欢迎提交Pull Requests或Issues!

## 致谢

本项目受到[GPT-InvestAR](https://github.com/UditGupta10/GPT-InvestAR)的启发，并在此基础上进行了多项改进和创新。

感谢CSC6052/5051/4100/DDA6307/MDS5110 NLP课程的支持。

## 许可证

[MIT License](LICENSE)
