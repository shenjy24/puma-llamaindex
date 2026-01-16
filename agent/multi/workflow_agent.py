import os
import asyncio
from dotenv import load_dotenv
from llama_index.tools.tavily_research import TavilyToolSpec
from llama_index.llms.deepseek import DeepSeek
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
from llama_index.core.agent.workflow import (
    AgentOutput,
    ToolCall,
    ToolCallResult,
)

# 加载 .env 文件中的环境变量
# 这行代码会查找当前目录下的 .env 文件并将变量注入到 os.environ 中
load_dotenv()

# 从环境变量获取配置, 建议设置默认值作为兜底，防止 .env 漏配
api_key = os.getenv("DEEPSEEK_API_KEY")
api_base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
tavily_api_key = os.getenv("TAVILY_API_KEY")

llm = DeepSeek(
    model=model_name,
    api_base=api_base,
    api_key=api_key,
    temperature=0.1,  # Agent 任务建议低温度
    max_retries=3,
)

tavily_tool = TavilyToolSpec(api_key=tavily_api_key)
search_web = tavily_tool.to_tool_list()[0]


async def record_notes(ctx: Context, notes: str, notes_title: str) -> str:
    """Useful for recording notes on a given topic."""
    current_state = await ctx.get("state")
    if "research_notes" not in current_state:
        current_state["research_notes"] = {}
    current_state["research_notes"][notes_title] = notes
    await ctx.set("state", current_state)
    return "Notes recorded."


async def write_report(ctx: Context, report_content: str) -> str:
    """Useful for writing a report on a given topic."""
    current_state = await ctx.get("state")
    current_state["report_content"] = report_content
    await ctx.set("state", current_state)
    return "Report written."


async def review_report(ctx: Context, review: str) -> str:
    """Useful for reviewing a report and providing feedback."""
    current_state = await ctx.get("state")
    current_state["review"] = review
    await ctx.set("state", current_state)
    return "Report reviewed."


# create our specialist agents
research_agent = FunctionAgent(
    name="ResearchAgent",
    description="Search the web and record notes.",
    system_prompt="You are a researcher… hand off to WriteAgent when ready.",
    llm=llm,
    tools=[search_web, record_notes],
    can_handoff_to=["WriteAgent"],
)

write_agent = FunctionAgent(
    name="WriteAgent",
    description="Writes a markdown report from the notes.",
    system_prompt="You are a writer… ask ReviewAgent for feedback when done.",
    llm=llm,
    tools=[write_report],
    can_handoff_to=["ReviewAgent", "ResearchAgent"],
)

review_agent = FunctionAgent(
    name="ReviewAgent",
    description="Reviews a report and gives feedback.",
    system_prompt="You are a reviewer…",
    llm=llm,
    tools=[review_report],
    can_handoff_to=["WriteAgent"],
)

# wire them together
agent_workflow = AgentWorkflow(
    agents=[research_agent, write_agent, review_agent],
    root_agent=research_agent.name,
    initial_state={
        "research_notes": {},
        "report_content": "Not written yet.",
        "review": "Review required.",
    },
)


async def main():
    handler = agent_workflow.run(
        user_msg="""
        Write me a report on the history of the web. Briefly describe the history 
        of the world wide web, including the development of the internet and the 
        development of the web, including 21st century developments.
    """
    )

    current_agent = None
    current_tool_calls = ""
    async for event in handler.stream_events():
        if (
            hasattr(event, "current_agent_name")
            and event.current_agent_name != current_agent
        ):
            current_agent = event.current_agent_name
            print(f"\n{'='*50}")
            print(f"🤖 Agent: {current_agent}")
            print(f"{'='*50}\n")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("📤 Output:", event.response.content)
            # if event.tool_calls:
            #     print(
            #         "🛠️  Planning to use tools:",
            #         [call.tool_name for call in event.tool_calls],
            #     )
        # elif isinstance(event, ToolCallResult):
        #     print(f"🔧 Tool Result ({event.tool_name}):")
        #     print(f"  Arguments: {event.tool_kwargs}")
        #     print(f"  Output: {event.tool_output}")
        # elif isinstance(event, ToolCall):
        #     print(f"🔨 Calling Tool: {event.tool_name}")
        #     print(f"  With arguments: {event.tool_kwargs}")


if __name__ == "__main__":
    asyncio.run(main())
