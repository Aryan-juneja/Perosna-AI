# flake8: noqa
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
import requests



load_dotenv()
llm = init_chat_model(model_provider="openai", model="gpt-4.1")
class State(TypedDict):
    messages: Annotated[list, add_messages]
workflow = StateGraph(State)

@tool()
def get_weather(city: str):
    """This tool returns the weather data about the given city"""

    url = f"https://wttr.in/{city.lower()}?format=%C+%t"
    response = requests.get(url)

    if response.status_code == 200:
        return f"The weather in {city} is {response.text}."

    return "Something went wrong"

llm_with_tools = llm.bind_tools([get_weather])

def chatbot(state: State):
    """This is a simple chatbot that can answer questions and use tools."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}
tools=[get_weather]
tool_node = ToolNode(tools=tools)

workflow.add_node("chatbot",chatbot)
workflow.add_node("tools", tool_node)
workflow.add_edge(START, "chatbot")
workflow.add_conditional_edges(
    "chatbot",
    tools_condition,
)
workflow.add_edge("tools", "chatbot")

def Checkpointsaver(checkpointer):
    graph = workflow.compile(checkpointer=checkpointer)
    return graph

def main():
    DB_URI = "mongodb://admin:admin@mongodb:27017"
    config = {"configurable": {"thread_id": "1"}}
    with MongoDBSaver.from_conn_string(DB_URI) as checkpointer:
        graph = Checkpointsaver(checkpointer)
        while True:
            user_input = input("> ")
            state = State(
                messages=[{"role": "user", "content": user_input}]
            )
            for event in graph.stream(state, config, stream_mode="values"):
                if "messages" in event:
                    event["messages"][-1].pretty_print()
                    
main()