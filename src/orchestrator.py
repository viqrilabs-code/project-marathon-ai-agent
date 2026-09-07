import os, sys
from pydantic import BaseModel
from fastapi import FastAPI
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import START, END, MessagesState, StateGraph
from typing import List, Any, Dict, Optional
from langchain_google_genai import ChatGoogleGenerativeAI

class AgentState(BaseModel):
    messages: str
    ops: Optional[str]
    a: Optional[float]
    b: Optional[float]
    result: Optional[float]

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
app = FastAPI(title="Calculator Orchestrator", description="Orchestrates calculator operations based on user input.")

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


async def predict_ops(state:AgentState) -> AgentState:
    model = ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_MODEL", "gemini-3.6-flash")
    )
    prompt = f"Based on the following messages, determine the operation to perform: {state.messages}. The operation should be one of the following: add, subtract, multiply, divide. If no operation is needed, respond with 'none'."
    operation = await model.ainvoke(prompt)
    print(f"Predicted operation: {operation.content[0]['text']}")
    state.ops = operation.content[0]['text']    
    return  state

def route(state: AgentState):
    return state.ops

builder = StateGraph(AgentState)
builder.add_node("predict_ops", predict_ops)
builder.add_node("add", add)
builder.add_node("subtract", subtract)
builder.add_node("multiply", multiply)
builder.add_node("divide", divide)

builder.add_edge(START, "predict_ops")
builder.add_conditional_edges("predict_ops", route, {
    "add": "add",
    "subtract": "subtract",
    "multiply": "multiply",
    "divide": "divide",
    "none": END
})  
builder.add_edge("add", END)
builder.add_edge("subtract", END)
builder.add_edge("multiply", END)
builder.add_edge("divide", END)     

@app.post("/calculate")
async def calculate(state: AgentState):
    graph = builder.compile()
    result = await graph.ainvoke(state)
    print("="*30)
    l = len(f"FINAL RESULT: {result['result']}")
    left = (30-l)//2
    right = 30 - l - left
    output = "="*left + f" FINAL RESULT: {result['result']} "+"="*right
    print(output[0:30])
    print("="*30)
    return {"result": result}
