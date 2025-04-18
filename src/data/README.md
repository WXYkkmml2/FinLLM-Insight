# 数据获取模块

本模块负责从公开数据源获取中国上市公司的年度报告，并提供数据格式转换和处理功能。

## 功能模块

### 1. 年报下载 (`download_reports.py`)

通过AKShare API从巨潮资讯网获取中国上市公司的年度报告，支持以下功能：

- 获取指定指数（如沪深300）成分股列表
- 自动下载年报PDF文件
- 按年份分类存储文件
- 生成下载结果报告

#### 使用方法

```bash
# 使用默认配置文件
python src/data/download_reports.py

# 指定配置文件
python src/data/download_reports.py --config_path path/to/config.json
```

#### 配置项说明

在`config/config.json`中可以设置以下参数：

- `annual_reports_html_save_directory`: 年报保存的根目录
- `china_stock_index`: 股票指数名称，默认"CSI300"（沪深300）
- `min_year`: 最早下载的年份，默认2018年
- `download_delay`: 下载间隔时间（秒），防止请求过于频繁

### 2. 格式转换 (`convert_formats.py`)

将下载的PDF文件转换为后续处理所需的格式，实现：

- PDF转文本
- 中文文本预处理
- 繁体转简体（如需）

### 3. 目标变量生成 (`make_targets.py`)

基于股票历史价格数据生成模型训练所需的目标变量：

- 获取股票历史价格数据
- 计算不同时间窗口的收益率
- 生成分类标签（如"上涨"、"下跌"）
## 数据获取与使用

1. 运行`src/data/download_reports.py`下载年报
2. 运行`src/data/convert_formats.py`将PDF转换为文本
3. 运行`src/data/make_targets.py`生成目标变量

## 数据说明

### 文件结构

数据将按以下结构组织：
本目录包含FinLLM-Insight项目的所有数据文件，按照处理阶段分为原始数据和处理后数据。

## 目录结构

```
data/
├── raw/                 # 原始数据
│   └── annual_reports/  # 原始年报PDF文件
│       ├── 2018/        # 按年份分类存储的年报
│       ├── 2019/
│       └── ...
│       └── download_results.csv  # 下载结果记录
└── processed/           # 处理后的数据
    ├── text_reports/    # 转换为文本的年报
    │   ├── 2018/
    │   ├── 2019/
    │   └── ...
    │   └── conversion_results.csv  # 转换结果记录
    ├── targets/         # 模型训练的目标变量
    │   ├── stock_targets.csv  # 所有股票的目标变量数据
    │   └── visualizations/    # 目标变量的可视化图表
    ├── embeddings/      # 文本嵌入向量
    └── features/        # LLM生成的特征数据
```



### 原始数据 (raw)

- **annual_reports/**: 包含从巨潮资讯网下载的上市公司年度报告PDF文件
  - 文件命名格式：`股票代码_年份_annual_report.pdf`
  - 按照年份分类存储
  - `download_results.csv`记录下载状态和文件路径

### 处理后数据 (processed)

- **text_reports/**: 从PDF转换提取的文本文件
  - 文件命名格式：`股票代码_年份_annual_report.txt`
  - 文本已进行预处理（繁体转简体、标点规范化等）
  - `conversion_results.csv`记录转换状态和文件路径

- **targets/**: 基于股票历史价格生成的目标变量
  - `stock_targets.csv`包含每个股票在年报发布后不同时间窗口的收益率数据
  - 包含1天、5天、20天、60天和120天的未来收益率
  - 同时提供二分类标签（上涨/下跌）和四分类标签（极差/较差/较好/优秀）

- **embeddings/**: 存储文本嵌入向量
  - 使用中文语言模型生成的文档片段嵌入向量
  - 使用ChromaDB存储，方便检索

- **features/**: LLM生成的结构化特征
  - 基于大语言模型对年报内容的分析结果
  - 包含财务状况、经营风险、发展前景等多维度评分
  ### 注意事项

- 数据下载需要稳定的网络连接
- 遵循巨潮资讯网的使用规范，设置合理的下载间隔时间
- 部分报告可能无法自动下载，请查看日志文件了解详情
- 原始PDF文件较大，请确保有足够的磁盘空间
- 数据下载和处理可能需要较长时间，请耐心等待
- 请遵循相关数据源的使用条款和限制



