import os
# 必须放在第一行，任何其他 import 之前
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import chromadb
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.llms.deepseek import DeepSeek
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 加载 .env 文件中的环境变量
# 这行代码会查找当前目录下的 .env 文件并将变量注入到 os.environ 中
load_dotenv()

# 从环境变量获取配置, 建议设置默认值作为兜底，防止 .env 漏配
api_key = os.getenv("DEEPSEEK_API_KEY")
api_base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# Settings control global defaults
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.llm = DeepSeek(
    model=model_name,
    api_base=api_base,
    api_key=api_key,
    temperature=0.1,  # Agent 任务建议低温度
    max_retries=3,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

async def init(): 
    # load some documents
    documents = SimpleDirectoryReader(DATA_DIR).load_data()

    # initialize client, setting path to save data
    db = chromadb.PersistentClient(path=CHROMA_DIR)

    # create collection
    chroma_collection = db.get_or_create_collection("quickstart")

    # assign chroma as the vector_store to the context
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # create your index
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context
    )

    # create a query engine and query
    query_engine = index.as_query_engine()
    response = query_engine.query("What is the meaning of life?")
    print(response)

# 文档已经存入向量数据库，读取即可
async def load(): 

    # initialize client, setting path to save data
    db = chromadb.PersistentClient(path=CHROMA_DIR)

    # create collection
    chroma_collection = db.get_or_create_collection("quickstart")

    # assign chroma as the vector_store to the context
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # create your index
    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context
    )

    # create a query engine and query
    query_engine = index.as_query_engine()
    response = query_engine.query("What did the author do in college?")
    print(response)

if __name__ == "__main__":
    import asyncio
    asyncio.run(load())