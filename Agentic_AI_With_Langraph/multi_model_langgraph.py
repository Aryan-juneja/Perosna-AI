from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
from openai import OpenAI
from typing_extensions import TypedDict
from typing import Optional

# OpenAI and Gemini client setup
openai_client = OpenAI()
client = OpenAI(
    api_key="GEMINI_API_KEY",  # Caution: Keep keys secret in real code
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Pydantic model for Gemini's response
class AccuracyOutput(BaseModel):
    accuracy: str

# LangGraph state schema
class State(TypedDict):
    query: str
    result: Optional[str]
    isQuestion_orCode: Optional[bool]
    accuracy: Optional[str]
    recheck: Optional[int]

# Model to parse classification (question vs code)
class IsQuestionOrCode(BaseModel):
    isQuestion_orCode: bool

# Step 1: Check if query is question or code-related
def is_question_or_code(state: State) -> State:
    response = openai_client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Just return true if the input is a question and false if it is a code-related task."},
            {"role": "user", "content": state["query"]}
        ]
    )
    content = response.choices[0].message.content.strip().lower()
    state["isQuestion_orCode"] = "true" in content
    return state

# Step 2a: For general questions
def chat_with_llm(state: State) -> State:
    response = openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": state["query"]}
        ]
    )
    state["result"] = response.choices[0].message.content
    return state

# Step 2b: For code-related queries
def code_with_llm(state: State) -> State:
    response = openai_client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": state["query"]}
        ]
    )
    state["result"] = response.choices[0].message.content
    return state

# Step 3: Recheck accuracy with Gemini
def recheck_with_llm(state: State) -> State:
    query = state["query"]
    llm_code = state["result"]

    # Using string formatting since Gemini expects template-like messages
    prompt = f"""You are an expert at evaluating code accuracy. Given the user query and the code output, return the accuracy as a percentage string.

User Query:
{query}

Code Output:
{llm_code}

Return ONLY the accuracy in percentage (e.g., "85%")."""

    response = client.chat.completions.create(
        model="gemini-1.5-flash",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    state["accuracy"] = response.choices[0].message.content.strip()
    return state

# Building the graph
graph_builder = StateGraph(State)

# Add decision node first
graph_builder.add_node("is_question_or_code", is_question_or_code)

# Add branches
graph_builder.add_node("chat_with_llm", chat_with_llm)
graph_builder.add_node("code_with_llm", code_with_llm)
graph_builder.add_node("recheck_with_llm", recheck_with_llm)

# Transitions
graph_builder.add_edge(START, "is_question_or_code")

# Conditional branching based on question/code type
def route_based_on_type(state: State) -> str:
    return "chat_with_llm" if state["isQuestion_orCode"] else "code_with_llm"

graph_builder.add_conditional_edges("is_question_or_code", route_based_on_type)

# Common post-processing
graph_builder.add_edge("chat_with_llm", END)
graph_builder.add_edge("code_with_llm", "recheck_with_llm")
graph_builder.add_edge("recheck_with_llm", END)

# Compile the graph
graph = graph_builder.compile()

# Runner
def main():
    input_query = input("Enter your query: ")
    initial_state: State = {
        "query": input_query,
        "result": None,
        "isQuestion_orCode": None,
        "accuracy": None,
        "recheck": None
    }
    final_state = graph.invoke(initial_state)
    print("\nFinal Result:")
    print("Response:", final_state["result"])
    print("Accuracy:", final_state["accuracy"])

if __name__ == "__main__":
    main()
