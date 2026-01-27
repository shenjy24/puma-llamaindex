import os
from typing import List, Optional

# 生产级别的RAG服务示例

# LlamaIndex v0.10+ 核心组件
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    StorageContext,
    SimpleDirectoryReader,
    Document,
    get_response_synthesizer,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank

# 向量库与嵌入模型
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
import qdrant_client


class ProductionRAGService:
    def __init__(
        self,
        collection_name: str = "production_rag_hybrid",
        qdrant_url: str = "http://localhost:6333",
        openai_api_key: str = None,
    ):
        """
        初始化 RAG 服务，配置混合检索与重排序
        """
        # 1. 全局模型配置 (使用 v0.10+ Settings)
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key

        Settings.llm = OpenAI(model="gpt-4o", temperature=0.1)
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

        # 2. 文本切分策略 (Chunking)
        # 生产环境建议：块大一些以保留上下文，但在检索时切分更细或使用 overlap
        Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

        # 3. 初始化 Qdrant 客户端 (生产级向量数据库)
        # 注意：enable_hybrid=True 开启稀疏向量索引，fastembed_sparse_model 指定稀疏模型
        self.client = qdrant_client.QdrantClient(url=qdrant_url)
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            enable_hybrid=True,
            fastembed_sparse_model="Qdrant/bm25",  # 使用轻量级 BM25 模型生成稀疏向量
        )

        # 4. 初始化重排序模型 (Re-ranker)
        # Cross-Encoder 比单纯向量相似度更精准，但计算较慢，用于第二阶段筛选
        # 也可以使用 CohereRerank (需 API Key)
        self.reranker = SentenceTransformerRerank(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            top_n=3,  # 最终给 LLM 的上下文数量
        )

        self.index = self._load_or_create_index()

    def _load_or_create_index(self) -> VectorStoreIndex:
        """加载现有索引或创建新索引的容器"""
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        # 尝试从向量库加载索引 (不重新计算 embedding)
        try:
            index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store, storage_context=storage_context
            )
            print("✅ 已连接到现有的持久化向量索引。")
            return index
        except Exception as e:
            print(f"ℹ️ 初始化空索引: {e}")
            return VectorStoreIndex.from_documents([], storage_context=storage_context)

    def ingest_documents(self, data_dir: str):
        """
        数据摄入管道：读取 -> 切分 -> 嵌入 -> 存储
        """
        print(f"📂 正在从 {data_dir} 读取文档...")
        documents = SimpleDirectoryReader(data_dir).load_data()

        # 使用 IngestionPipeline 处理去重和转换
        pipeline = IngestionPipeline(
            transformations=[
                Settings.node_parser,
                Settings.embed_model,
            ],
            vector_store=self.vector_store,
        )

        # 运行管道 (计算 embedding 并存入 Qdrant)
        # 这一步会自动计算 Dense Vector (OpenAI) 和 Sparse Vector (BM25)
        nodes = pipeline.run(documents=documents)
        print(f"🎉 成功索引 {len(nodes)} 个节点到 Qdrant。")

    def query(self, query_text: str) -> str:
        """
        执行 RAG 查询：混合检索 -> 重排序 -> LLM 合成
        """
        # 1. 配置混合检索器 (Hybrid Retriever)
        # alpha 参数控制权重：0.5 表示 50% 向量搜索 + 50% 关键词搜索
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=10,  # 召回更多文档用于重排序 (例如 10 个)
            vector_store_query_mode="hybrid",
            sparse_top_k=10,
            alpha=0.5,
        )

        # 2. 构建查询引擎
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=[self.reranker],  # 在此处加入重排序
            response_synthesizer=get_response_synthesizer(response_mode="compact"),
        )

        # 3. 执行查询
        response = query_engine.query(query_text)

        # (可选) 打印检索到的来源以供调试
        # for node in response.source_nodes:
        #     print(f"Debug Source: {node.score:.4f} - {node.text[:50]}...")

        return str(response)


# --- 使用示例 ---
if __name__ == "__main__":
    # 假设你已经在本地启动了 Qdrant (docker run -p 6333:6333 qdrant/qdrant)

    rag_service = ProductionRAGService(
        openai_api_key="sk-proj-...",  # 替换为你的 Key
        qdrant_url="http://localhost:6333",
    )

    # 1. 首次运行时摄入数据
    # rag_service.ingest_documents("./data")

    # 2. 提问
    answer = rag_service.query("LlamaIndex 的混合检索是如何工作的？")
    print(f"\n🤖 回答:\n{answer}")
