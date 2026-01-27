import os

# 必须放在第一行，任何其他 import 之前
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import asyncio
from dotenv import load_dotenv
from llama_index.llms.deepseek import DeepSeek

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

# Create a RAG tool using LlamaIndex
documents = SimpleDirectoryReader("rag/data").load_data()
index = VectorStoreIndex.from_documents(
    documents,
    # we can optionally override the embed_model here
    # embed_model=Settings.embed_model,
)

query_engine = index.as_query_engine(
    # we can optionally override the llm here
    # llm=Settings.llm,
)


async def search_documents(query: str) -> str:
    """Useful for answering natural language questions about an personal essay written by Paul Graham."""
    response = await query_engine.aquery(query)
    return str(response)


# Create an enhanced workflow with both tools
agent = AgentWorkflow.from_tools_or_functions(
    [search_documents],
    llm=Settings.llm,
    system_prompt="""You are a helpful assistant that can search through documents to answer questions.""",
)


# Now we can ask questions about the documents or do calculations
async def main():
    response = await agent.run("What did the author do in college?")
    print(response)


# Run the agent
if __name__ == "__main__":
    asyncio.run(main())
