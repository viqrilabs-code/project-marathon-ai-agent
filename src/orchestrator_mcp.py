import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel


load_dotenv(Path(__file__).resolve().parent.parent / ".env")

app = FastAPI()


class AskRequest(BaseModel):
    message: str


async def build_graph():
    client = MultiServerMCPClient(
        {
            "calculator": {
                "transport": "http",
                "url": "http://127.0.0.1:8001/mcp",
            }
        }
    )
    tools = await client.get_tools()

    model = ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
    ).bind_tools(tools)

    async def call_model(state: MessagesState):
        response = await model.ainvoke(state["messages"])
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")

    return builder.compile()


@app.post("/ask")
async def ask(request: AskRequest):
    graph = await build_graph()
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=request.message)]}
    )

    return {"answer": result["messages"][-1].content}
