# 特征工程模块

本模块负责从处理后的年报文本中提取特征，包括生成文本嵌入向量和使用大语言模型进行结构化特征提取。

## 功能模块

### 1. 嵌入向量生成 (`embeddings.py`)

将年报文本转换为向量表示，便于后续的相似度检索和内容理解，支持以下功能：

- 文本分块：将长文本分割成适当大小的片段
- 向量化：使用预训练的中文语言模型生成文本嵌入
- 向量存储：利用ChromaDB存储和索引嵌入向量
- 元数据管理：为每个文本片段关联公司代码、年份等元数据

#### 使用方法

```bash
# 使用默认配置文件
python src/features/embeddings.py

# 指定配置文件
python src/features/embeddings.py --config_path path/to/config.json
```

#### 配置项说明

在`config/config.json`中可以设置以下参数：

- `processed_reports_text_directory`: 处理后文本文件的目录
- `embeddings_directory`: 嵌入向量存储目录
- `embedding_model`: 使用的嵌入模型名称，如"BAAI/bge-large-zh-v1.5"
- `chunk_size`: 文本分块的大小
- `
