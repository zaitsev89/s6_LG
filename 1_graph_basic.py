# ===== IMPORTS =====
from datetime import datetime
from typing import Annotated
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from prompts import SYSTEM_PROMPT
from typing_extensions import TypedDict

# Load environment variables
load_dotenv()

# ===== LLM SETUP =====
llm = ChatOpenAI(
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY")
)


# ===== STATE DEFINITION =====
class State(TypedDict):
    messages: Annotated[list, add_messages]


# ===== NODE DEFINITIONS =====
def call_model_node(state: State):
    system_message = SYSTEM_PROMPT.format(
        mood="neutral",
        system_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    message = llm.invoke([{
        "role": "system",
        "content": system_message
    }, *state["messages"]])

    return {"messages": [message]}


# ===== GRAPH CONSTRUCTION =====
graph_builder = StateGraph(State)

graph_builder.add_node("call_model", call_model_node)

graph_builder.add_edge(START, "call_model")
graph_builder.add_edge("call_model", END)

graph = graph_builder.compile()
