# ===== IMPORTS =====
import os
from typing import Annotated

from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatPerplexity
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

memory = MemorySaver()


# ===== STATE DEFINITION =====
class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]
    user_name: str


# ===== LLM SETUP =====
# Initialize the main LLM (Claude)
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")


# ===== TOOL DEFINITION =====
@tool(parse_docstring=True)
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
def human_assistance(query: str, user_name: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]


# Register tools and bind to LLM
tools = [internet_search, human_assistance]
llm_with_tools = llm.bind_tools(tools)


# ===== NODE DEFINITIONS =====
# Main chatbot node - processes messages with the LLM
def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    # Because we will be interrupting during tool execution,
    # we disable parallel tool calling to avoid repeating any
    # tool invocations when we resume.
    assert len(message.tool_calls) <= 1
    return {"messages": [message]}


# Router function to determine if tool execution is needed
def route_tools(state: State):
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
# Initialize the graph builder
graph_builder = StateGraph(State)

# Create tool execution node
tool_node = ToolNode(tools)

# Add nodes to the graph
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

# Define the flow of the graph
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    {"tools": "tools", END: END},
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("chatbot", END)

# Compile the graph
graph = graph_builder.compile(checkpointer=memory)


# ===== VISUALIZATION (OPTIONAL) =====
try:
    os.makedirs("./ignore-utils", exist_ok=True)
    with open("./ignore-utils/quickstart3.png", "wb") as f:
        f.write(graph.get_graph(xray=True).draw_mermaid_png())
except Exception:
    # This requires some extra dependencies and is optional
    pass


# ===== EXECUTION FUNCTIONS =====
def stream_graph_updates(user_input: str):
    """Process user input through the graph and stream the results."""

    config = {"configurable": {"thread_id": "1"}}
    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}], "user_name": "Pedro"},
        config,
        stream_mode="values",
    )

    for event in events:
        event["messages"][-1].pretty_print()
        # Check if the message has tool calls and needs human assistance
        if (
            hasattr(event["messages"][-1], "tool_calls")
            and event["messages"][-1].tool_calls
            and event["messages"][-1].tool_calls[0].get("name") == "human_assistance"
        ):
            print("Human assistance needed!")
            human_response = input("Human: ")
            if human_response.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            human_command = Command(resume={"data": human_response})
            events = graph.stream(human_command, config, stream_mode="values")
            for event in events:
                if "messages" in event:
                    event["messages"][-1].pretty_print()


# ===== MAIN INTERACTION LOOP =====
while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break
