import os
import openai
import asyncio
from dotenv import load_dotenv
from typing import List, Any
import qdrant_client
from qdrant_client import models, AsyncQdrantClient, QdrantClient
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.memory import ChatSummaryMemoryBuffer
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.embeddings import BaseEmbedding

load_dotenv()

# --- 1. 基础配置 (建议放环境变量) ---
silicon_api_key = os.getenv("SILICONFLOW_API_KEY", "")
silicon_api_base = os.getenv("SILICONFLOW_API_BASE", "")
qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
qdrant_api_base: str = os.getenv("QDRANT_API_BASE", "")


# --- 2. 模拟数据库 (存储对话文本) ---
class ChatHistoryDB:
    @staticmethod
    def get_recent_messages(user_id: str) -> List[ChatMessage]:
        # 模拟：从数据库查询该用户的最近 10 条对话
        # 生产环境请使用 async 驱动连接 MySQL/MongoDB
        return [
            ChatMessage(role=MessageRole.USER, content="I love drinking latte."),
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content="Latte is great! Do you like it with sugar?",
            ),
        ]


# --- 3. 核心 Agent 管理器 ---
class EnglishCoachManager:
    def __init__(self):
        self.client = QdrantClient(url=qdrant_api_base, api_key=qdrant_api_key)
        self.aclient = AsyncQdrantClient(url=qdrant_api_base, api_key=qdrant_api_key)

        # 检查并自动创建 Collection
        self.collection_name = "english_app_memory"
        self._ensure_collection_exists()

        Settings.llm = OpenAILike(
            model="deepseek-ai/DeepSeek-V3",
            api_base=silicon_api_base,
            api_key=silicon_api_key,
            is_chat_model=True,
            # --- RAG 核心优化参数 (保持不变) ---
            # 1. 温度: 口语对话建议稍微带点随机性，更有趣
            temperature=0.7,
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
        Settings.embed_model = SiliconFlowEmbedding(
            model_name="BAAI/bge-m3",
            api_key=silicon_api_key,
            api_base=silicon_api_base,
        )

    def _get_user_vector_store(self, user_id: str):
        """实现 RAG 层的用户隔离"""
        return QdrantVectorStore(
            client=self.client,
            aclient=self.aclient,
            collection_name=self.collection_name,
            # 关键：强制每次检索都带上 user_id 过滤条件
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="user_id", match=models.MatchValue(value=user_id)
                    )
                ]
            ),
        )

    def get_chat_engine(self, user_id: str):
        # 加载长期记忆 (RAG 索引)
        vector_store = self._get_user_vector_store(user_id)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # 即使索引为空，也会返回一个可用的 index 对象
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

        # 加载短期记忆 (对话上下文)
        past_messages = ChatHistoryDB.get_recent_messages(user_id)
        memory = ChatSummaryMemoryBuffer.from_defaults(
            llm=Settings.llm,
            chat_history=past_messages,
            token_limit=2500,  # 超过则自动摘要
        )

        # 设定 Alex 的性格背景
        system_prompt = (
            "You are Alex, a professional and witty English coach.\n"
            "Personality: Encouraging, uses natural idioms, slightly humorous.\n"
            "Rules: Correct user's grammar briefly at the end of each turn. "
            "Refer to their past interests if found in the context."
        )

        # 构建 CondensePlusContextChatEngine
        # 这是最适合对话的模式：它会重写查询（Condense）并注入 RAG 事实（Context）
        return CondensePlusContextChatEngine.from_defaults(
            retriever=index.as_retriever(similarity_top_k=3),
            memory=memory,
            system_prompt=system_prompt,
            verbose=True,  # 控制台可见逻辑链路
        )

    def _ensure_collection_exists(self):
        """如果 Collection 不存在，则按正确配置创建它"""
        from qdrant_client.models import Distance, VectorParams

        # 检查是否存在
        if not self.client.collection_exists(self.collection_name):
            print(f"Creating collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=1024,  # BAAI/bge-m3 的标准输出维度
                    distance=Distance.COSINE,  # 英语口语/文本检索建议使用余弦距离
                ),
            )


class SiliconFlowEmbedding(BaseEmbedding):
    """
    专门为硅基流动 (SiliconFlow) 定制的 Embedding 类
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
        self._client = openai.Client(api_key=api_key, base_url=api_base)

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._get_embedding(text)

    def _get_embedding(self, text: str) -> List[float]:
        text = text.replace("\n", " ")
        response = self._client.embeddings.create(input=[text], model=self._model_name)
        return response.data[0].embedding

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)


async def main():
    # 单例初始化
    manager = EnglishCoachManager()

    # 1. 获取针对用户 ID 为 "1" 的引擎
    user_id = "1"
    engine = manager.get_chat_engine(user_id)

    # 2. 对话执行
    print(f"--- Starting conversation for user: {user_id} ---")
    user_input = "What is my favorite drink based on our history?"
    response = await engine.achat(user_input)

    # 3. 结果输出
    output = {
        "user_input": user_input,
        "reply": response.response,
        "sources": [node.node.get_content() for node in response.source_nodes],
    }

    print("\n[AI Response]:")
    print(output["reply"])
    if output["sources"]:
        print(f"\n[RAG Sources]: {output['sources']}")


if __name__ == "__main__":
    asyncio.run(main())
