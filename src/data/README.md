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

## 数据说明

### 文件结构

数据将按以下结构组织：

```
data/
├── raw/
│   └── annual_reports/
│       ├── 2018/
│       │   ├── 000001_2018_annual_report.pdf
│       │   └── ...
│       ├── 2019/
│       └── ...
│       └── download_results.csv
└── processed/
    ├── embeddings/
    ├── features/
    └── targets/
```

### 注意事项

- 数据下载需要稳定的网络连接
- 遵循巨潮资讯网的使用规范，设置合理的下载间隔时间
- 部分报告可能无法自动下载，请查看日志文件了解详情
