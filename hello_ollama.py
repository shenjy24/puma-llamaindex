import asyncio
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.ollama import Ollama

# Define a simple calculator tool
def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b

# Create an agent workflow with our calculator tool
agent = FunctionAgent(
    tools=[multiply],
    llm=Ollama(model="llama3.1", base_url="http://localhost:11434", request_timeout=360.0, 
               # Manually set the context window to limit memory usage
               context_window=8000),
    system_prompt="You are a helpful assistant that can multiply two numbers."
)

async def main():
    # Run the agent
    response = await agent.run("What is 12.5 * 4?")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())