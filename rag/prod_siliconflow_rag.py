# 生产级别的RAG服务示例: 使用硅基流动的模型
import os
from dotenv import load_dotenv
from typing import Any, List

# LlamaIndex v0.10+ 核心组件
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    StorageContext,
    SimpleDirectoryReader,
    get_response_synthesizer,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client

import requests
from typing import List, Optional
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle

import openai
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer

# 加载 .env 文件中的环境变量
# 这行代码会查找当前目录下的 .env 文件并将变量注入到 os.environ 中
load_dotenv()


class ProductionRAGService:
    def __init__(
        self,
        collection_name: str = "production_rag_hybrid",
        qdrant_url: str = "http://localhost:6333",
        qdrant_api_key: str = None,
        silicon_api_key: str = None,
        silicon_api_base: str = None,
    ):
        """
        初始化 RAG 服务，配置混合检索与重排序
        """
        # 1. 全局模型配置 (使用 v0.10+ Settings)
        # =========================================================
        # 配置 Embedding (嵌入模型) - 使用硅基流动云端版 BGE-M3
        # =========================================================
        # 原来的 HuggingFaceEmbedding 是本地跑，现在改用 OpenAIEmbedding 调云端 API
        # 硅基流动支持的模型 ID 为: "BAAI/bge-m3"
        Settings.embed_model = SiliconFlowEmbedding(
            model_name="BAAI/bge-m3",
            api_key=silicon_api_key,
            api_base=silicon_api_base,
        )

        # =========================================================
        # 配置 LLM (大语言模型) - 使用硅基流动 DeepSeek-V3
        # =========================================================
        # 注意：硅基流动的 DeepSeek V3 模型 ID 通常是 "deepseek-ai/DeepSeek-V3"
        # 如果你想用 R1，就改成 "deepseek-ai/DeepSeek-R1"
        Settings.llm = OpenAILike(
            model="deepseek-ai/DeepSeek-V3",
            api_base=silicon_api_base,
            api_key=silicon_api_key,
            is_chat_model=True,
            # --- RAG 核心优化参数 (保持不变) ---
            # 1. 温度: 极低，减少幻觉
            temperature=0.0,
            # 2. 上下文窗口: 即使是 OpenAI 类，最好也显式声明，防止库默认使用 GPT-3.5 的 4k 限制
            # 告诉 LlamaIndex 这个模型能吃 60k token
            context_window=60000,
            # 3. 最大输出
            max_tokens=4096,
            # 4. 重试机制
            max_retries=3,
            # 5. 额外参数
            additional_kwargs={
                "top_p": 0.95,
            },
            # 6. (可选) 设为 True 可以让 LlamaIndex 复用 API 连接，提升一点点速度
            reuse_client=True,
        )

        # 2. 文本切分策略 (Chunking)
        # 生产环境建议：块大一些以保留上下文，但在检索时切分更细或使用 overlap
        Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

        # 3. 初始化 Qdrant 客户端 (生产级向量数据库)
        # 注意：enable_hybrid=True 开启稀疏向量索引，fastembed_sparse_model 指定稀疏模型
        self.client = qdrant_client.QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            enable_hybrid=True,
            fastembed_sparse_model="Qdrant/bm25",  # 使用轻量级 BM25 模型生成稀疏向量
        )

        # 4. 初始化重排序模型 (Re-ranker)
        # ==========================================
        # 使用方法：替换掉你原来的 SentenceTransformerRerank
        # ==========================================
        # 硅基流动目前支持的模型 ID 是 "BAAI/bge-reranker-v2-m3"
        self.reranker = SiliconFlowRerank(
            model="BAAI/bge-reranker-v2-m3",
            api_key=silicon_api_key,
            top_n=3,
        )

        # 记忆组件
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=3900)

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

    def stream_query(self, query_text: str):
        """
        执行流式 RAG 查询：混合检索 -> 重排序 -> LLM 流式合成
        返回一个生成器，逐个 token 输出响应
        """
        # 立即给用户反馈（非常重要）
        yield "🔍 正在检索相关资料...\n"

        # 配置混合检索器 (Hybrid Retriever)
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=10,
            vector_store_query_mode="hybrid",
            sparse_top_k=10,
            alpha=0.5,
        )

        # 构建查询引擎 (使用流式合成器)
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=[self.reranker],
            response_synthesizer=get_response_synthesizer(
                response_mode="compact", streaming=True
            ),
        )

        yield "🧠 正在生成答案...\n\n"

        # 执行流式查询
        try:
            response = query_engine.query(query_text)
        except Exception as e:
            yield f"⚠️ 查询失败：{str(e)}"
            return

        # 稳定流式输出
        if hasattr(response, "response_gen") and response.response_gen:
            for token in response.response_gen:
                yield token
        else:
            # 降级兜底（极少发生）
            yield str(response)

    def stream_memory_query(self, query_text: str):
        """
        执行带记忆功能的流式 RAG 查询
        """
        # 立即给用户反馈（非常重要）
        yield "🔍 正在检索相关资料...\n"

        # 配置混合检索器 (Hybrid Retriever)
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=10,
            vector_store_query_mode="hybrid",
            sparse_top_k=10,
            alpha=0.5,
        )

        # 构建 ContextChatEngine
        # 它会自动处理：历史对话重写 + 知识检索 + 答案合成
        chat_engine = ContextChatEngine.from_defaults(
            retriever=retriever,
            node_postprocessors=[self.reranker],
            memory=self.memory,
            system_prompt="你是一个专业助手。请结合给定的本地知识库和对话历史来回答问题。",
        )

        yield "🧠 正在生成答案...\n\n"

        # 执行流式查询
        try:
            response = chat_engine.stream_chat(query_text)
        except Exception as e:
            yield f"⚠️ 查询失败：{str(e)}"
            return

        # 稳定流式输出
        if hasattr(response, "response_gen") and response.response_gen:
            for token in response.response_gen:
                yield token
        else:
            # 降级兜底（极少发生）
            yield str(response)


class SiliconFlowEmbedding(BaseEmbedding):
    """
    专门为硅基流动 (SiliconFlow) 定制的 Embedding 类
    绕过 LlamaIndex 对 OpenAI 模型名称的强制校验
    """

    _client: openai.Client = PrivateAttr()
    _model_name: str = PrivateAttr()

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        api_key: str = None,
        api_base: str = "https://api.siliconflow.cn/v1",
        **kwargs: Any,
    ):
        super().__init__(model_name=model_name, **kwargs)
        self._model_name = model_name
        # 初始化标准的 OpenAI 客户端
        self._client = openai.Client(api_key=api_key, base_url=api_base)

    def _get_query_embedding(self, query: str) -> List[float]:
        """获取单个查询的 embedding"""
        return self._get_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        """获取单个文档片段的 embedding"""
        return self._get_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        批量获取 embedding (关键优化：减少网络请求次数)
        """
        # 移除换行符是 embedding 的最佳实践
        texts = [t.replace("\n", " ") for t in texts]
        try:
            response = self._client.embeddings.create(
                input=texts, model=self._model_name
            )
            # 按照返回顺序提取 embedding
            return [data.embedding for data in response.data]
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            raise e

    def _get_embedding(self, text: str) -> List[float]:
        """内部通用方法"""
        text = text.replace("\n", " ")
        response = self._client.embeddings.create(input=[text], model=self._model_name)
        return response.data[0].embedding

    async def _aget_query_embedding(self, query: str) -> List[float]:
        # 简单起见，暂不实现异步，直接调同步方法
        return self._get_query_embedding(query)


class SiliconFlowRerank(BaseNodePostprocessor):
    """
    自定义的硅基流动 Rerank 处理器
    """

    siliconflow_api_base: str = os.getenv("SILICONFLOW_API_BASE", "")
    model: str = Field(description="Rerank model name")
    top_n: int = Field(description="Top N nodes to return")
    api_key: str = Field(description="SiliconFlow API Key")
    base_url: str = Field(default=siliconflow_api_base, description="API Endpoint")

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if not nodes:
            return []

        request_url = self.base_url
        if not request_url.endswith("/rerank"):
            # 简单的拼接处理，防止用户只填了 base 域名
            if request_url.endswith("/v1"):
                request_url = f"{request_url}/rerank"
            else:
                request_url = f"{request_url}/v1/rerank"

        # 准备请求数据
        documents = [node.node.get_content() for node in nodes]
        payload = {
            "model": self.model,
            "query": query_bundle.query_str,
            "documents": documents,
            "top_n": self.top_n,
            "return_documents": False,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # 发送请求
        try:
            response = requests.post(request_url, json=payload, headers=headers)
            response.raise_for_status()
            results = response.json().get("results", [])

            # 根据返回的 index 重新排序并赋值分数
            new_nodes = []
            for res in results:
                idx = res["index"]
                score = res["relevance_score"]

                node = nodes[idx]
                node.score = score  # 更新分数为 Cross-Encoder 的精准分数
                new_nodes.append(node)

            return new_nodes

        except Exception as e:
            print(f"Rerank API Error: {e}")
            # 如果 API 挂了，降级返回原来的前 N 个，防止程序崩溃
            return nodes[: self.top_n]


# --- 使用示例 ---
if __name__ == "__main__":

    silicon_api_key: str = os.getenv("SILICONFLOW_API_KEY", "")
    silicon_api_base = os.getenv("SILICONFLOW_API_BASE", "")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    qdrant_api_base: str = os.getenv("QDRANT_API_BASE", "")

    rag_service = ProductionRAGService(
        qdrant_url=qdrant_api_base,
        qdrant_api_key=qdrant_api_key,
        silicon_api_key=silicon_api_key,
        silicon_api_base=silicon_api_base,
    )

    # 1. 首次运行时摄入数据
    # base_dir = os.path.dirname(os.path.abspath(__file__))
    # data_dir = os.path.join(base_dir, "data")
    # rag_service.ingest_documents(data_dir)

    # 2. 流式提问示例
    stream_generator = rag_service.stream_query(
        "PDFBox 提供的一些关键功能和功能有哪些？"
    )
    for token in stream_generator:
        print(token, end="", flush=True)
    print("\n")

    # 3. 非流式提问示例 (如需要)
    # answer = rag_service.query("PDFBox 提供的一些关键功能和功能有哪些？")
    # print(f"\n🤖 回答:\n{answer}")
