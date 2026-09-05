import asyncio
import os
import sys
from pathlib import Path
from typing import Any, List, Dict, Optional
from pydantic import BaseModel

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import START, MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition



PROJECT_ROOT = Path(__file__).resolve().parent.parent
CALCULATOR_SERVER = PROJECT_ROOT / "mcp-server" / "calc_server.py"

class state(BaseModel):
    messages: List[Any] = []

async def build_orc() -> Any:
    client = MultiServerMCPClient(
        {
            "calculator": {
                "transport": "http",
                "url": "http://127.0.0.1:8001/mcp",
        }
        }
    )

    tools = await client.get_tools()
    print(f"Tools: {tools}")

    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, max_output_tokens=512)

    model_with_tools = model.bind_tools(tools)

    async def run_model(state:state) -> state:
        response = await model_with_tools.ainvoke(state.messages)
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    graph_builder = StateGraph(state)
    graph_builder.add_node("agent", run_model)
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_edge(START,"agent")
    graph_builder.add_conditional_edges(
        "agent",
        tools_condition,
    )

    return graph_builder.compile()


async def main():
    load_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "Add GOOGLE_API_KEY to the project .env file."
        )

    graph = await build_orc()

    return await graph.ainvoke({"messages": [HumanMessage(content="What is 2 + 2?")]})


if __name__ == "__main__":
    result = asyncio.run(main())
    print(f"Result: {result}")
