##calculator
# Query: I need to do summation on the given two numbers 5, 7.

 #- LLM function: Decide which calculator operation to perform based on user input. -> OUtput: Opseration name (add, subtract, multiply, divide) or 'none' if no operation is needed.
 #- Routing function: This will route the graph execution to appropiate function. -> Output: Rsult of the operation
 #- Simply return the output

#=====================================================Dummy agent====================================================================================================================
import asyncio
import os,sys
from dotenv import load_dotenv
from pathlib import Path
from fastapi import FastAPI
from langchain_google_genai  import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END, MessagesState
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

app = FastAPI(title="Calculator")

class AgentState(BaseModel):
    messages: str
    ops: str
    a: float
    b: float
    result: str

def add(state: AgentState) -> AgentState:
    state.result = state.a + state.b
    return state

def subtract(state: AgentState) -> AgentState:
    state.result = state.a - state.b
    return state            


def multiply(state: AgentState) -> AgentState:
    state.result = state.a * state.b
    return state            


def divide(state: AgentState) -> AgentState:
    if state.b == 0:
        raise ValueError("Cannot divide by zero.")
    state.result = state.a / state.b
    return state

async def ops_classifcation(state:AgentState) -> AgentState:
    model = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        api_key=os.getenv("GOOGLE_API_KEY")
    )
    prompt = f"Based on the following messages, Perform the operation: {state.messages}."
    res = await model.ainvoke(prompt)
    print((f"Predicted operation: {res.content[0]['text']}"))
    state.result = res.content[0]['text']
    return  state

#START -> ops_classifcation -> add -> END

builder = StateGraph(AgentState)
builder.add_node("operation-name", ops_classifcation)
builder.add_node("add", add)
builder.add_node("subtract", subtract)
builder.add_node("multiply", multiply)
builder.add_node("divide", divide)


builder.add_edge(START, "operation-name")
builder.add_edge("operation-name", "add")
builder.add_edge("operation-name", "subtract")
builder.add_edge("operation-name", "multiply")
builder.add_edge("operation-name", "divide")
        
builder.add_edge("add", END)

@app.post("/calculate")
async def main(state: AgentState):
    graph = builder.compile()

    response = await graph.ainvoke(state)
    print(f"Final Result: {response}" )
    return {"result": response}

# if __name__ == "__main__":
#     result = asyncio.run(main())
#     print(f"Result: {result}")


