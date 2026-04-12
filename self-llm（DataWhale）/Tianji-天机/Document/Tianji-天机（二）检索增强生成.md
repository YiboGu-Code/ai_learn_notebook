# Tianji-天机（二）检索增强生成

## **1. RAG (检索增强生成) 系统**

RAG (Retrieval-Augmented Generation) 是一种将检索系统与大语言模型结合的技术，通过检索相关知识来增强模型的回答能力。RAG 系统通过将大语言模型与外部知识库相结合,实现了知识的实时更新和精确控制。

相比传统的模型微调方法，RAG 具有显著的优势:它无需重新训练模型就能够更新知识库内容,可以精确控制模型使用的知识来源,能够清晰地追踪模型回答的知识来源,并且在成本效益方面比微调更具优势。这种技术已经在多个重要场景中得到了广泛应用,包括企业知识库问答系统的构建、智能客服系统的开发、海量文档的检索与总结,以及个性化助手服务的实现等。

在实际应用中,RAG 技术面临着几个关键性的技术挑战需要解决:

1. 如何设计合理的文档分块策略,在保持文档语义完整性的同时实现高效的检索;
2. 检索质量的优化问题,需要不断提高系统对相关文档的召回率和准确率;
3. 上下文融合的问题,即如何有效地将多个检索到的文档片段信息进行整合;
4. 答案生成的质量问题,系统需要基于检索到的内容生成准确、连贯且符合上下文的回答。

Tianji 在 RAG 技术上进行了深入探索,提供了一套完整的入门demo。我们基于 LangChain 和 LlamaIndex 两大主流框架,实现了包括文档处理、知识库构建、检索优化等全流程的功能。

### **1.1 Tianji 的 RAG 实现思路**

#### **1.1.1 核心特点**

1. 多框架支持：
    - 同时支持 LangChain 和 LlamaIndex 两大主流框架
    - 提供不同复杂度的实现示例，方便学习和使用
2. 全流程优化：
    - 数据预处理：使用 LLM 辅助数据清洗和负样本生成
    - 知识库构建：支持多种数据源，包括本地文件和网页内容
    - 检索增强：实现了重排序、文档扩展等优化技术
    - 场景定制：针对不同业务场景提供专门的知识库和处理流程
3. 工程最佳实践：
    - 模块化设计：各组件高度解耦，便于扩展和维护
    - 性能优化：使用 FAISS 向量索引，支持大规模知识库
    - 易用性：提供完整的命令行工具和示例代码

#### **1.1.2 扩展功能**

1. 高级检索技术：
    - 混合检索：结合关键词和语义检索
    - 重排序：对检索结果进行二次排序
    - 上下文扩展：自动扩展相关文档片段
2. 数据处理增强：
    - 智能分块：基于语义的文档分块策略
    - 数据清洗：自动识别和过滤低质量内容
    - 知识聚类：使用 K-means 优化知识组织

### **1.2 入门指南**

#### **1.2.1 什么是 RAG？**

RAG 系统主要包含以下核心组件：

1. 文档加载器：从不同来源加载文档
2. 文本分割器：将文档分割成适当大小的块
3. 向量数据库：存储文档块的向量表示
4. 检索器：根据查询检索相关文档
5. 大语言模型：结合检索到的上下文生成回答

#### **1.2.2. 基本工作流程**

1. 准备阶段：
    - 加载文档
    - 文档分块
    - 生成嵌入向量
    - 存储到向量数据库
2. 查询阶段：
    - 接收用户问题
    - 检索相关文档
    - 结合上下文生成回答

### **1.3 实现方式**

#### **1.3.1. LangChain 实现**

在 Tianji 项目中，我们提供了基于 LangChain 的 RAG 实现示例：

```python
# 基本组件
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# 核心流程
1. 加载文档
loader = DirectoryLoader(data_path, glob="*.txt")

2. 文本分割
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=200
)

3. 创建向量数据库
vectordb = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding_func,
    persist_directory=persist_directory,
)
```

#### **1.3.2. LlamaIndex 实现**

我们同时提供了基于 LlamaIndex 的实现方式：

```python
# 基本组件
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.vector_stores.faiss import FaissVectorStore

# 核心流程
1. 加载文档
documents = SimpleDirectoryReader(data_dir).load_data()

2. 创建索引
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)

3. 创建检索器
retriever = VectorIndexRetriever(index=index)
```

### **1.4. Tianji 项目中的测试文件**

您可以通过以下测试文件学习 RAG 的实现：

1. 核心实现：
    - `run/demo_rag_langchain_all.py`: 完整的多场景 RAG 系统示例，基于 LangChain
    - `test/knowledges/llamaindex/test_RAG_zhipuai_simple.py`: 基于 LlamaIndex 的简单 RAG 实现
    - `test/knowledges/llamaindex/test_RAG_zhipuai_advanced.py`: 高级 RAG 实现，包含重排序等功能
2. 数据处理工具：
    - `tools/rag/0-data_llm_filter.py`: 使用 LLM 过滤和清洗训练数据
    - `tools/rag/0-data_llm_filter_negative.py`: 生成负样本数据
    - `tools/rag/0-data_llm_filter_lesswords.py`: 处理短文本数据
    - `tools/rag/1-get_rag_knowledges.py`: 知识库构建工具
    - `tools/rag/2-jsonknowledges_kmeans.py`: 使用 K-means 聚类处理知识库
    - `tools/rag/3-json2txt.py`: JSON 格式转换为文本格式
3. 文档处理工具：
    - `tools/rag/article2chunk..js`: 文章分块工具
    - `tools/rag/url2article.md`: 网页内容抓取指南

这些文件涵盖了 RAG 系统的完整流程：

- 数据收集和清洗
- 知识库构建和优化
- 文本分块和向量化
- 检索和问答实现
- 高级功能（如重排序、聚类等）