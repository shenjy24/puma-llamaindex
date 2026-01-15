import os
import time
import asyncio
from dotenv import load_dotenv
from llama_index.llms.deepseek import DeepSeek
from llama_index.core.agent.workflow import AgentWorkflow

# 加载 .env 文件中的环境变量
# 这行代码会查找当前目录下的 .env 文件并将变量注入到 os.environ 中
load_dotenv()

# 从环境变量获取配置, 建议设置默认值作为兜底，防止 .env 漏配
api_key = os.getenv("DEEPSEEK_API_KEY")
api_base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# 检查 key 是否存在，避免运行时报错不直观
if not api_key:
    raise ValueError("未找到 API Key，请在 .env 文件中配置 DEEPSEEK_API_KEY")

def multiply(a: float, b: float) -> float:
    """Multiply two numbers and returns the product"""
    return a * b

def add(a: float, b: float) -> float:
    """Add two numbers and returns the sum"""
    return a + b

llm = DeepSeek(
    model=model_name,
    api_base=api_base,
    api_key=api_key,
    temperature=0.1,  # Agent 任务建议低温度
    max_retries=3
)

workflow = AgentWorkflow.from_tools_or_functions(
    [multiply, add],
    llm=llm,
    system_prompt="You are an agent that can perform basic mathematical operations using tools."
)

async def main():
    response = await workflow.run(user_msg="What is 20+(2*4)?")
    print(response)

if __name__ == "__main__":
    # 记录开始时间
    start_time = time.perf_counter()
    
    # 运行主程序
    asyncio.run(main())
    
    # 记录结束时间并计算差值
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    
    print(f"\n执行耗时: {elapsed_time:.4f} 秒")