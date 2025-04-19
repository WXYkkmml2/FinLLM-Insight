# 特征工程模块

## 环境配置

### 确保 ChromaDB 版本支持异步 API

要确保你的 ChromaDB 版本支持异步 API，需要遵循以下步骤：

1. **检查 ChromaDB 版本**：
   ```bash
   pip show chromadb
   ```
   确保版本至少为 0.4.0 或更高，因为较新版本才完全支持异步 API。

2. **更新 ChromaDB**（如果需要）：
   ```bash
   pip install --upgrade chromadb
   ```

3. **验证异步 API 可用**：
   创建一个简单的测试脚本来验证异步 API 是否可用：
   ```python
   import asyncio
   import chromadb

   async def test_async_api():
       try:
           client = chromadb.AsyncClient()
           print("异步 API 可用!")
           return True
       except (AttributeError, ImportError) as e:
           print(f"错误: {e}")
           print("当前 ChromaDB 版本不支持异步 API")
           return False

   if __name__ == "__main__":
       result = asyncio.run(test_async_api())
       print(f"测试结果: {'成功' if result else '失败'}")
   ```

4. **检查依赖关系**：
   确保安装了支持异步操作的相关依赖，例如：
   ```bash
   pip install aiohttp
   ```

5. **查阅 ChromaDB 文档**：
   随时查阅最新的 [ChromaDB 文档](https://docs.trychroma.com/)，了解异步 API 的变更和推荐用法。

如果环境中 ChromaDB 不支持异步 API（例如使用了较旧版本），可以使用以下替代方案：

1. 继续使用同步 API，但通过多线程或进程池来并行处理多个文件
2. 将异步代码包装在兼容性层中，在不支持异步时自动回退到同步操作

注意：使用异步 API 可以提高性能，特别是在处理大量文档时。这种方法可以更有效地利用 I/O 等待时间，从而加快整体处理速度。

## 功能模块

本模块负责从处理后的年报文本中提取特征，包括生成文本嵌入向量和使用大语言模型进行结构化特征提取。

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
- `chunk_overlap`: 文本分块的重叠大小

### 2. LLM特征生成 (`llm_features.py`)

利用大语言模型对年报内容进行分析，生成结构化的评分和投资建议特征，支持以下功能：

- 检索相关文本：基于问题从向量数据库中检索相关内容
- LLM分析：使用大语言模型对检索到的内容进行分析
- 特征提取：从LLM回答中提取结构化特征，如评分和分类
- 结果存储：将特征和原始回答保存为CSV和JSON文件

#### 使用方法

```bash
# 使用默认配置文件和问题
python src/features/llm_features.py

# 指定配置文件和问题文件
python src/features/llm_features.py --config_path path/to/config.json --questions_path path/to/questions.json
```

#### 配置项说明

在`config/config.json`中可以设置以下参数：

- `embeddings_directory`: 嵌入向量的存储目录
- `features_directory`: 特征输出目录
- `llm_model`: 使用的大语言模型名称，如"gpt-3.5-turbo-16k"
- `max_tokens_per_call`: 每次LLM调用的最大令牌数

在`config/questions.json`中可以设置分析问题，包括：

- 财务健康状况评分
- 商业模式和竞争优势评分
- 未来增长潜力评分
- 管理层质量评分
- 风险评估
- 行业前景评分
- ESG表现评分
- 创新能力评分
- 投资建议分类

## 数据流向

本模块的数据流向如下：

```
文本文件 -> 文本分块 -> 嵌入生成 -> 向量存储 -> 相关内容检索 -> LLM分析 -> 特征提取 -> 特征文件
```

## 注意事项

- 嵌入向量生成需要足够的计算资源，尤其是处理大量文本时
- LLM API调用需要OpenAI或其他LLM服务的API密钥
- LLM API调用可能需要较长时间，请耐心等待
- 特征生成过程中会产生API调用费用，请合理控制调用次数和频率
