# ===== IMPORTS =====
from datetime import datetime
from typing import Annotated
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatPerplexity
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from prompts import SYSTEM_PROMPT
from typing_extensions import TypedDict

# Load environment variables
load_dotenv()

# ===== LLM SETUP =====
llm = ChatOpenAI(
    model="gpt-4-turbo",
    api_key=os.getenv("OPENAI_API_KEY")
)

# ===== MEMORY =====
memory = MemorySaver()


# ===== STATE DEFINITION =====
class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    mood: str = "Neutral"


# ===== TOOL DEFINITION =====
@tool
def internet_search(question: str) -> dict | str:
    """Search the internet for information about the given question.

    Args:
        question: The question to search for information about.

    Returns:
        Information about the question.
    """
    # Initialize Perplexity model for internet search
    perplexity_chat_model_sonar = ChatPerplexity(
        model="sonar",
        temperature=0,
        api_key=os.getenv("PERPLEXITY_API_KEY")
    )

    # Define prompt for Perplexity
    PERPLEXITY_PROMPT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an helpful assistant, searching the internet for information to answer the user's question.",
            ),
            ("human", "{question}"),
        ]
    )

    # Create and execute the chain
    chain = PERPLEXITY_PROMPT | perplexity_chat_model_sonar
    answer = chain.invoke({"question": question})
    return answer


@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]


@tool
def change_mood(mood: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> str:
    """Change the mood of the agent.
    Args:
        mood: The mood to change to.
        tool_call_id: The tool call ID.
    """
    state_update = {
        "mood": mood,
        "messages": [ToolMessage(f"Mood changed to {mood}", tool_call_id=tool_call_id)],
    }
    return Command(update=state_update)


tools = [internet_search, human_assistance, change_mood]


# ===== NODE DEFINITIONS =====
def call_model_node(state: State):
    system_message = SYSTEM_PROMPT.format(
        mood=state.get("mood", "neutral"),
        system_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    llm_with_tools = llm.bind_tools(tools)

    message = llm_with_tools.invoke(
        [{"role": "system", "content": system_message}, *state["messages"]]
    )

    assert len(message.tool_calls) <= 1

    return {"messages": [message]}


tool_node = ToolNode(tools)


# ===== ROUTERS =====
def tools_router(state: State):
    """Use in the conditional_edge to route to the ToolNode if the last message has tool calls.
    Otherwise, route to the end."""
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")

    # Check if the message contains tool calls
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


# ===== GRAPH CONSTRUCTION =====
graph_builder = StateGraph(State)

graph_builder.add_node("call_model", call_model_node)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "call_model")
graph_builder.add_edge("tools", "call_model")
graph_builder.add_edge("call_model", END)

graph_builder.add_conditional_edges(
    "call_model",
    tools_router,
    {"tools": "tools", END: END},
)

graph = graph_builder.compile(checkpointer=memory)
