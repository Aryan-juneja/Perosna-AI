from langgraph.graph import StateGraph, START , END

from openai import OpenAI
from typing_extensions import TypedDict
openai_client = OpenAI()


class State(TypedDict):
    query:str
    result:str | None

def chat_with_llm(state:State):
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": state["query"]}
        ]
    )
    state["result"] = response.choices[0].message.content
    return state

graph_builder = StateGraph(State)

graph_builder.add_node("chat_with_llm",chat_with_llm)
graph_builder.add_edge(START,"chat_with_llm")
graph_builder.add_edge("chat_with_llm", END)

graph =graph_builder.compile()

def main():
    input_query = input("Enter your query: ")
    initial_state = State(query=input_query, result=None)
    final_state = graph.invoke(initial_state)
    print("Response from LLM:", final_state)

main()
