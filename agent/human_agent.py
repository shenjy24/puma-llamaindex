import os
import time
import asyncio
from dotenv import load_dotenv
from llama_index.llms.deepseek import DeepSeek
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.workflow import Context
from llama_index.core.workflow import (
    InputRequiredEvent,
    HumanResponseEvent,
)


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


# a tool that performs a dangerous task
async def dangerous_task(ctx: Context) -> str:
    """A dangerous task that requires human confirmation."""

    # emit an event to the external stream to be captured
    ctx.write_event_to_stream(
        InputRequiredEvent(
            prefix="Are you sure you want to proceed? ",
            user_name="Laurie",
        )
    )

    # wait until we see a HumanResponseEvent
    response = await ctx.wait_for_event(
        HumanResponseEvent, requirements={"user_name": "Laurie"}
    )

    # act on the input from the event
    if response.response.strip().lower() == "yes":
        return "Dangerous task completed successfully."
    else:
        return "Dangerous task aborted."


llm = DeepSeek(
    model=model_name,
    api_base=api_base,
    api_key=api_key,
    temperature=0.1,  # Agent 任务建议低温度
    max_retries=3,
)

workflow = AgentWorkflow.from_tools_or_functions(
    [dangerous_task],
    llm=llm,
    system_prompt="""You are an execution agent.
    You MUST call the provided tools when the user requests an action.
    Do NOT refuse or explain safety policies.
    Human confirmation is handled inside the tool, not by you.
    """,
)


async def main():
    handler = workflow.run(user_msg="I want to proceed with the dangerous task.")

    async for event in handler.stream_events():
        # capture InputRequiredEvent
        if isinstance(event, InputRequiredEvent):
            # capture keyboard input
            response = input(event.prefix)
            # send our response back
            handler.ctx.send_event(
                HumanResponseEvent(
                    response=response,
                    user_name=event.user_name,
                )
            )

    response = await handler
    print(str(response))


if __name__ == "__main__":
    # 记录开始时间
    start_time = time.perf_counter()

    # 运行主程序
    asyncio.run(main())

    # 记录结束时间并计算差值
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print(f"\n执行耗时: {elapsed_time:.4f} 秒")
