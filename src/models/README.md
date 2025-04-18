# 模型训练与预测模块

本模块负责基于LLM特征训练机器学习模型，并使用训练好的模型进行股票收益率预测。

## 功能模块

### 1. 模型训练 (`train.py`)

使用从年报分析中提取的特征训练机器学习模型，支持以下功能：

- 特征数据加载与预处理
- 模型训练与交叉验证
- 特征重要性分析
- 模型评估与可视化
- 模型保存与元数据记录

#### 使用方法

```bash
# 使用默认配置训练随机森林回归模型
python src/models/train.py

# 指定模型类型和算法
python src/models/train.py --model_type regression --model_name random_forest

# 指定预测时间窗口
python src/models/train.py --target_window 20
```

#### 支持的模型类型

- **回归模型 (regression)**：预测具体收益率
  - `linear`: 线性回归
  - `random_forest`: 随机森林回归器
  - `gradient_boosting`: 梯度提升回归器

- **分类模型 (classification)**：预测涨跌方向
  - `logistic`: 逻辑回归
  - `random_forest`: 随机森林分类器

### 2. 模型预测 (`predict.py`)

使用训练好的模型对新数据进行预测，支持以下功能：

- 加载最新或指定的模型
- 生成股票收益率或涨跌预测
- 预测结果可视化
- 生成投资建议报告

#### 使用方法

```bash
# 使用最新的回归模型进行预测
python src/models/predict.py

# 指定模型文件和类型
python src/models/predict.py --model_path ./models/your_model.pkl --info_path ./models/your_model_info.json

# 指定预测时间窗口和模型类型
python src/models/predict.py --model_type classification --target_window 60

# 不生成HTML摘要报告
python src/models/predict.py --no_summary
```

## 模型输出说明

### 训练输出

训练过程会在`models`目录下生成以下文件：

- **模型文件**：`{model_type}_{model_name}_{target_window}d_{timestamp}.pkl`
- **模型信息**：`{model_type}_{model_name}_{target_window}d_{timestamp}_info.json`
- **特征重要性**：`feature_importance.csv`
- **可视化图表**：`visualizations/feature_importance.png`、`visualizations/predicted_vs_actual.png`等

### 预测输出

预测过程会在`predictions`目录下生成以下文件：

- **预测结果**：`predictions_{timestamp}.csv`
- **预测摘要**：`prediction_summary_{timestamp}.html`
- **可视化图表**：`visualizations/prediction_distribution_{timestamp}.png`等

## 预测结果解释

回归模型的预测结果包含以下字段：

- `company_code`：公司代码
- `report_year`：报告年份
- `predicted_return_{N}d`：预测的N天收益率
- `prediction_percentile`：收益率预测的百分位数
- `prediction_category`：收益率预测的分类（很可能下跌、可能小幅下跌、可能小幅上涨、很可能上涨）

分类模型的预测结果包
