import os
import asyncio
from urllib import response
from dotenv import load_dotenv
from llama_index.llms.deepseek import DeepSeek
from llama_index.core.agent.workflow import FunctionAgent
from pydantic import BaseModel, Field
from typing import List
import json
from llama_index.core.llms import ChatMessage
from typing import List, Dict, Any

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
    max_retries=3,
)


def get_flavor(ice_cream_shop: str):
    return "Strawberry with no extra sugar"


# define structured output format and tools
class Flavor(BaseModel):
    flavor: str
    with_sugar: bool


# 结构化输出函数
async def structured_output_parsing(
    messages: List[ChatMessage],
) -> Dict[str, Any]:
    sllm = llm.as_structured_llm(Flavor)
    messages.append(
        ChatMessage(
            role="user",
            content="Given the previous message history, structure the output based on the provided format.",
        )
    )
    response = await sllm.achat(messages)
    return json.loads(response.message.content)


agent = FunctionAgent(
    tools=[get_flavor],
    name="ice_cream_shopper",
    system_prompt="You are an agent that knows the ice cream flavors in various shops.",
    structured_output_fn=structured_output_parsing,
    llm=llm,
)


async def main():
    response = await agent.run("What strawberry flavor is available at Gelato Italia?")
    print(response.structured_response)
    print(response.get_pydantic_model(Flavor))


if __name__ == "__main__":
    asyncio.run(main())
