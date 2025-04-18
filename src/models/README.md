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

分类模型的预测结果包含以下字段：

- `company_code`：公司代码
- `report_year`：报告年份
- `predicted_up_{N}d`：预测的N天涨跌方向（1表示上涨，0表示下跌）
- `prediction_confidence`：预测的置信度（如果可用）

## 模型评估指标

### 回归模型评估指标

- **MSE (均方误差)**：预测值与实际值差异的平方平均值
- **RMSE (均方根误差)**：MSE的平方根，与原始值单位相同
- **MAE (平均绝对误差)**：预测值与实际值绝对差异的平均值
- **R² (决定系数)**：模型解释的方差占总方差的比例，值越接近1越好

### 分类模型评估指标

- **Accuracy (准确率)**：正确预测的比例
- **Precision (精确率)**：预测为正例中真正例的比例
- **Recall (召回率)**：真正例中被正确预测的比例
- **F1 Score**：精确率和召回率的调和平均值

## 特征重要性分析

特征重要性分析可以帮助理解哪些因素对股票收益率预测最重要。对于随机森林和梯度提升模型，特征重要性基于特征对预测准确性的贡献来计算。对于线性模型，特征重要性基于系数的绝对值来计算。

特征重要性分析可以帮助：

- 识别最具预测力的LLM分析维度
- 优化特征选择和提问策略
- 理解模型的决策依据

## 注意事项

- 模型训练需要足够数量的样本，建议至少有100家公司的数据
- 模型性能受到数据质量的影响，确保LLM特征的质量和一致性
- 股市预测本质上存在不确定性，模型预测仅供参考，不构成投资建议
- 随着新数据的加入，应定期重新训练模型以保持预测准确性
- 对于重要投资决策，建议结合多种模型和其他分析方法
