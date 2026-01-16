import os
import asyncio
from dotenv import load_dotenv
from llama_index.llms.deepseek import DeepSeek
from llama_index.core.agent.workflow import FunctionAgent
from pydantic import BaseModel, Field

# 加载 .env 文件中的环境变量
# 这行代码会查找当前目录下的 .env 文件并将变量注入到 os.environ 中
load_dotenv()

# 从环境变量获取配置, 建议设置默认值作为兜底，防止 .env 漏配
api_key = os.getenv("DEEPSEEK_API_KEY")
api_base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

llm = DeepSeek(
    model=model_name,
    api_base=api_base,
    api_key=api_key,
    temperature=0.1,  # Agent 任务建议低温度
    max_retries=3
)

# define structured output format and tools
class MathResult(BaseModel):
    operation: str = Field(description="the performed operation")
    result: int = Field(description="the result of the operation")

def multiply(x: int, y: int):
    """Multiply two numbers"""
    return x * y

## define agent
agent = FunctionAgent(
    tools=[multiply],
    name="calculator",
    system_prompt="You are a calculator agent who can multiply two numbers using the `multiply` tool.",
    output_cls=MathResult,
    llm=llm,
)

async def main():
    response = await agent.run("What is 3415 * 43144?")
    print(response.structured_response)
    print(response.get_pydantic_model(MathResult))

if __name__ == "__main__":
    asyncio.run(main())