# FinLLM-Insight 系统源代码

本目录包含FinLLM-Insight项目的所有源代码文件，按照功能模块组织。

## 目录结构

```
src/
├── data/                  # 数据获取和处理模块
│   ├── download_reports.py    # 下载年报
│   ├── convert_formats.py     # 格式转换
│   └── make_targets.py        # 生成目标变量
├── features/              # 特征工程模块
│   ├── embeddings.py          # 生成文本嵌入
│   └── llm_features.py        # 生成LLM特征
├── models/                # 模型训练与预测模块
│   ├── train.py               # 模型训练
│   └── predict.py             # 模型预测
├── rag/                   # 检索增强生成模块
│   └── rag_component.py       # RAG系统
└── pipeline.py            # 完整处理流水线
```

## 系统流水线 (`pipeline.py`)

系统流水线模块可以一键执行从数据获取到模型预测的整个流程，实现了全流程自动化。

### 功能特点

- **完整流程** - 整合项目所有步骤，从数据获取到投资预测
- **灵活配置** - 通过配置文件和命令行参数控制执行流程
- **断点续传** - 支持跳过已完成的步骤，从中断处继续执行
- **并行执行** - 支持适用步骤的并行执行，提高效率
- **执行日志** - 详细记录执行过程，便于问题诊断
- **执行摘要** - 生成执行摘要，展示各步骤完成情况

### 使用方法

```bash
# 执行完整流水线
python src/pipeline.py

# 使用指定配置文件
python src/pipeline.py --config_path custom_config.json

# 跳过特定步骤
python src/pipeline.py --skip_steps 1,2

# 启用并行执行
python src/pipeline.py --parallel

# 仅执行特定步骤
python src/pipeline.py --step 3
```

### 流水线步骤

系统流水线包含以下步骤：

1. **下载年报数据** - 从数据源获取中国上市公司年度报告
2. **转换报告格式** - 将PDF报告转换为文本并进行预处理
3. **生成目标变量** - 基于股票历史价格生成预测目标
4. **生成文本嵌入** - 将文本转换为向量表示
5. **生成LLM特征** - 使用大语言模型分析年报内容生成特征
6. **训练预测模型** - 基于生成的特征训练机器学习模型
7. **生成预测结果** - 使用训练好的模型进行股票收益预测

### 配置选项

在`config/config.json`中可以设置以下主要参数：

- **目录配置** - 各类数据和模型的存储位置
- **模型参数** - 模型类型、算法选择和预测窗口
- **数据范围** - 报
