import os
import asyncio
from dotenv import load_dotenv
from llama_index.llms.deepseek import DeepSeek
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.tools.tavily_research import TavilyToolSpec
from llama_index.core.agent.workflow import (
    AgentInput,
    AgentOutput,
    ToolCallResult,
    AgentStream,
)

# 加载 .env 文件中的环境变量
# 这行代码会查找当前目录下的 .env 文件并将变量注入到 os.environ 中
load_dotenv()

# 从环境变量获取配置, 建议设置默认值作为兜底，防止 .env 漏配
api_key = os.getenv("DEEPSEEK_API_KEY")
api_base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
tavily_tool = TavilyToolSpec(api_key=os.getenv("TAVILY_API_KEY")) 

llm = DeepSeek(
    model=model_name,
    api_base=api_base,
    api_key=api_key,
    temperature=0.1,  # Agent 任务建议低温度
    max_retries=3
)

workflow = FunctionAgent(
    tools=tavily_tool.to_tool_list(),
    llm=llm,
    system_prompt="You're a helpful assistant that can search the web for information.",
)

async def main():
    handler = workflow.run(user_msg="What's the weather like in San Francisco?")

    # handle streaming output
    async for event in handler.stream_events():
        if isinstance(event, AgentStream):
            print(event.delta, end="", flush=True)
    #     elif isinstance(event, AgentInput):
    #         print("Agent input: ", event.input)  # the current input messages
    #         print("Agent name:", event.current_agent_name)  # the current agent name
    #     elif isinstance(event, AgentOutput):
    #         print("Agent output: ", event.response)  # the current full response
    #         print("Tool calls made: ", event.tool_calls)  # the selected tool calls, if any
    #         print("Raw LLM response: ", event.raw)  # the raw llm api response
    #     elif isinstance(event, ToolCallResult):
    #         print("Tool called: ", event.tool_name)  # the tool name
    #         print("Arguments to the tool: ", event.tool_kwargs)  # the tool kwargs
    #         print("Tool output: ", event.tool_output)  # the tool output            

    # # print final output
    # print(str(await handler))

if __name__ == "__main__":
    # 运行主程序
    asyncio.run(main())