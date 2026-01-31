import os
import openai
import asyncio
from dotenv import load_dotenv
from typing import List, Any
from llama_index.core.chat_engine import ContextChatEngine, SimpleChatEngine
from qdrant_client import models, AsyncQdrantClient, QdrantClient
from llama_index.core import VectorStoreIndex, StorageContext, Settings, Document
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.memory import ChatSummaryMemoryBuffer
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.response_synthesizers import get_response_synthesizer

load_dotenv()

# --- 1. 基础配置 (建议放环境变量) ---
silicon_api_key = os.getenv("SILICONFLOW_API_KEY", "")
silicon_api_base = os.getenv("SILICONFLOW_API_BASE", "")
qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
qdrant_api_base: str = os.getenv("QDRANT_API_BASE", "")


# --- 模拟数据库 (存储对话文本) ---
class ChatHistoryDB:
    @staticmethod
    def get_recent_messages(user_id: str) -> List[ChatMessage]:
        """获取用户最近的对话历史"""
        # 模拟：实际应从数据库查询
        # 注意：如果是新用户，返回空列表
        if user_id == "new_user":
            return []

        return [
            ChatMessage(role=MessageRole.USER, content="I love drinking latte."),
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content="Latte is great! Do you like it with sugar?",
            ),
        ]


# --- 核心 Agent 管理器 ---
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

        # Alex 的系统提示词
        self.system_prompt = (
            "You're Alex, an English coach who chats like a real friend texting.\n"
            "\n"
            "ABSOLUTE RULES:\n"
            "- Maximum 2-3 sentences total\n"
            "- NEVER use: parentheses, brackets, asterisks, PS, Note, BTW, FYI, Tip\n"
            "- NEVER use numbered or bulleted lists\n"
            "- NEVER add explanations after your main message\n"
            "- NO meta-commentary like 'let me explain' or 'here's what I mean'\n"
            "- If you correct grammar, weave it into your response naturally\n"
            "\n"
            "How to correct grammar naturally:\n"
            "✅ 'I remember you went to that cafe. Love the way you said \"went\" this time!'\n"
            '✅ \'Sounds fun! Just so you know, we say "I drank" not "I drunk".\'\n'
            '❌ \'That\'s great! PS: It should be "went" not "goed"\'\n'
            "❌ 'Nice! (By the way, the correct form is...)'\n"
            "\n"
            "BANNED phrases - never use these:\n"
            "- PS:\n"
            "- P.S.\n"
            "- Note:\n"
            "- BTW\n"
            "- By the way,\n"
            "- FYI\n"
            "- Just a tip:\n"
            "- Quick correction:\n"
            "- Side note:\n"
            "- Fun fact:\n"
            "\n"
            "Good examples:\n"
            "✅ 'Your accent is getting smoother! Try \"through\" one more time.'\n"
            "✅ 'I remember you love coffee. That cafe on Main Street has great lattes.'\n"
            '✅ \'That sounds awesome! We usually say "I went" instead of "I goed".\'\n'
            "\n"
            "Bad examples:\n"
            '❌ \'That sounds awesome! PS: It\'s "I went" not "I goed".\'\n'
            '❌ \'Great job! BTW, "dig" means "like" in slang.\'\n'
            "❌ 'Nice! (Just keeping it casual!)'\n"
            "❌ 'Cool story! Note: The past tense of go is went.'\n"
            "\n"
            "Just talk naturally like you're texting. That's it."
        )

    def get_chat_engine(self, user_id: str):
        """
        智能选择引擎：
        - 有长期记忆 → 使用 ContextChatEngine (RAG)
        - 无长期记忆 → 使用 SimpleChatEngine (纯对话)
        """
        # 加载短期记忆 (对话上下文)
        past_messages = ChatHistoryDB.get_recent_messages(user_id)

        # 创建 memory（即使消息为空也没关系）
        if past_messages:
            memory = ChatSummaryMemoryBuffer.from_defaults(
                llm=Settings.llm,
                chat_history=past_messages,
                token_limit=2500,
            )
        else:
            # 空 memory
            memory = ChatSummaryMemoryBuffer.from_defaults(
                llm=Settings.llm,
                chat_history=[],
                token_limit=2500,
            )

        # 加载长期记忆 (RAG 索引)
        has_memories = self._has_user_memories(user_id)
        if not has_memories:
            print(
                f"[INFO] User {user_id} has no long-term memories, using SimpleChatEngine"
            )
            # 返回简单对话引擎
            return SimpleChatEngine.from_defaults(
                llm=Settings.llm,
                memory=memory,
                system_prompt=self.system_prompt,
            )

        # 有长期记忆，使用 RAG 引擎
        print(f"[INFO] User {user_id} has long-term memories, using ContextChatEngine")

        vector_store = self._get_user_vector_store(user_id)
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

        # 优化的 context 模板
        context_template = (
            "Below are some relevant details from past conversations:\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Use this information naturally if relevant to the current conversation.\n"
            "If these facts are not relevant to the current user question, ignore them and chat freely.\n"
        )

        response_synthesizer = get_response_synthesizer(
            response_mode="compact",
            llm=Settings.llm,
        )

        # 构建 CondensePlusContextChatEngine
        # 这是最适合对话的模式：它会重写查询（Condense）并注入 RAG 事实（Context）
        return ContextChatEngine.from_defaults(
            retriever=index.as_retriever(similarity_top_k=3),
            memory=memory,
            system_prompt=self.system_prompt,
            response_synthesizer=response_synthesizer,  # 显式传入合成器
            context_template=context_template,
            verbose=True,
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

    def _ensure_collection_exists(self):
        """确保 Qdrant Collection 存在，并创建必要的索引"""
        from qdrant_client.models import Distance, VectorParams, PayloadSchemaType

        if not self.client.collection_exists(self.collection_name):
            print(f"Creating collection: {self.collection_name}")

            # 1. 创建 Collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=1024,
                    distance=Distance.COSINE,
                ),
            )

            # 2. 创建 user_id 索引（关键步骤）
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="user_id",
                field_schema=PayloadSchemaType.KEYWORD,  # 用于精确匹配
            )
            print(f"Created index for user_id field")
        else:
            # 如果 Collection 已存在，检查索引是否存在
            try:
                # 尝试获取 Collection 信息
                collection_info = self.client.get_collection(self.collection_name)

                # 检查 payload_schema 中是否有 user_id 索引
                if "user_id" not in collection_info.payload_schema:
                    print("Index for user_id not found, creating...")
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name="user_id",
                        field_schema=PayloadSchemaType.KEYWORD,
                    )
                    print("Created index for user_id field")
            except Exception as e:
                print(f"Warning: Could not verify index: {e}")

    def _has_user_memories(self, user_id: str) -> bool:
        """检查用户是否有长期记忆"""
        try:
            result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="user_id", match=models.MatchValue(value=user_id)
                        )
                    ]
                ),
                limit=1,
            )
            return len(result[0]) > 0
        except Exception as e:
            print(f"Error checking memories: {e}")
            return False

    async def add_memory(self, user_id: str, memory_text: str):
        """为用户添加长期记忆"""
        vector_store = self._get_user_vector_store(user_id)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # 创建文档，添加 metadata
        doc = Document(text=memory_text, metadata={"user_id": user_id})

        # 插入向量数据库
        index = VectorStoreIndex.from_documents(
            [doc],
            storage_context=storage_context,
        )
        print(f"[INFO] Added memory for user {user_id}: {memory_text[:50]}...")


class SiliconFlowEmbedding(BaseEmbedding):
    _client: openai.Client = PrivateAttr()
    _aclient: openai.AsyncClient = PrivateAttr()
    _model_name: str = PrivateAttr()

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        api_key: str = None,
        api_base: str = None,
        **kwargs: Any,
    ):
        super().__init__(model_name=model_name, **kwargs)
        self._model_name = model_name
        # 同时初始化同步和异步客户端，避免 asyncio.run 冲突
        self._client = openai.Client(api_key=api_key, base_url=api_base)
        self._aclient = openai.AsyncClient(api_key=api_key, base_url=api_base)

    def _get_query_embedding(self, query: str) -> List[float]:
        resp = self._client.embeddings.create(input=[query], model=self._model_name)
        return resp.data[0].embedding

    def _get_text_embedding(self, text: str) -> List[float]:
        resp = self._client.embeddings.create(input=[text], model=self._model_name)
        return resp.data[0].embedding

    async def _aget_query_embedding(self, query: str) -> List[float]:
        resp = await self._aclient.embeddings.create(
            input=[query], model=self._model_name
        )
        return resp.data[0].embedding


async def main():
    manager = EnglishCoachManager()

    # 测试 1: 新用户（无任何记忆）
    print("\n" + "=" * 60)
    print("TEST 1: New User (No Memories)")
    print("=" * 60)

    engine1 = manager.get_chat_engine("new_user")
    response1 = await engine1.achat("Hi! I'm new here. Can you introduce yourself?")
    print(f"\n[AI Response]: {response1.response}\n")

    # 测试 2: 有短期记忆的用户
    print("\n" + "=" * 60)
    print("TEST 2: User with Chat History")
    print("=" * 60)

    engine2 = manager.get_chat_engine("user_1")
    response2 = await engine2.achat(
        "What is my favorite drink? Tell me based on our chat history."
    )
    print(f"\n[AI Response]: {response2.response}\n")

    # 测试 3: 添加长期记忆后再测试
    print("\n" + "=" * 60)
    print("TEST 3: Adding Long-term Memory")
    print("=" * 60)

    await manager.add_memory(
        "user_1",
        "User mentioned they love latte with oat milk and no sugar. "
        "They also mentioned they are learning English to prepare for IELTS exam.",
    )

    # 重新获取引擎（这次会使用 ContextChatEngine）
    engine3 = manager.get_chat_engine("user_1")
    response3 = await engine3.achat("Can you recommend some coffee shops for me?")
    print(f"\n[AI Response]: {response3.response}\n")


if __name__ == "__main__":
    asyncio.run(main())
