# 检索增强生成 (RAG) 模块

本模块实现了基于年报内容的智能问答系统，结合检索增强生成 (Retrieval-Augmented Generation, RAG) 技术，为用户提供准确、全面的财报信息查询能力。

## 核心功能

- **智能问答**：通过自然语言提问获取年报中的信息
- **多公司查询**：支持查询单一或多个公司的年报内容
- **多年份查询**：支持指定年份的精确查询
- **上下文理解**：基于检索到的内容理解问题背景，提供有针对性的回答
- **信息溯源**：明确标识信息来源，便于用户查验

## 技术实现

FinancialReportRAG 类是RAG系统的核心实现，主要包括：

1. **向量检索**：利用预先计算的文本嵌入向量进行相似度检索
2. **相关性排序**：对检索结果进行排序，确保提供最相关的内容
3. **上下文构建**：将检索到的内容组织成结构化的上下文
4. **大语言模型生成**：利用LLM基于上下文生成针对性回答

## 使用方法

### 交互式命令行模式

```bash
# 启动交互式命令行界面
python src/rag/rag_component.py --interactive
```

在交互式模式中，可以通过以下格式提问：
- `问题内容` - 在所有公司年报中查找相关信息
- `公司代码 问题内容` - 在特定公司的所有年报中查找
- `公司代码 年份 问题内容` - 在特定公司特定年份的年报中查找

特殊命令：
- `companies` - 显示系统中所有可用的公司代码
- `exit` 或 `quit` - 退出程序

### API调用模式

```python
from src.rag.rag_component import FinancialReportRAG

# 初始化RAG系统
rag_system = FinancialReportRAG()

# 获取可用公司列表
companies = rag_system.get_available_companies()
print(f"系统中有 {len(companies)} 家公司")

# 获取特定公司的可用年份
years = rag_system.get_company_years("000001")
print(f"000001公司有以下年份的报告: {years}")

# 查询特定问题
result = rag_system.query(
    question="公司的主营业务是什么？",
    company_code="000001",
    year="2022"
)

# 输出结果
print(result["answer"])
print("信息来源:")
for source in result["sources"]:
    print(f"- {source['company_code']}公司 {source['year']}年 年报")
```

## 配置选项

在`config/config.json`中可以设置以下参数：

- `embeddings_directory`: 嵌入向量的存储目录
- `llm_model`: 使用的大语言模型，如"gpt-3.5-turbo-16k"
- `embedding_model`: 使用的嵌入模型，如"BAAI/bge-large-zh-v1.5"
- `max_tokens_per_call`: 每次LLM调用的最大令牌数

## 注意事项

- 系统依赖于预先生成的文本嵌入，请确保已运行`src/features/embeddings.py`
- LLM API调用需要相应的API密钥，请在环境变量中设置`OPENAI_API_KEY`
- 查询结果的质量取决于年报内容的完整性和嵌入向量的质量
- 复杂查询可能需要较长处理时间，尤其是在查询多家公司或大量文本时
