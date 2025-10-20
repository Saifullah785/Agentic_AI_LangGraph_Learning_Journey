# import necessary modules for building the state graph and chatbot
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv()


# Initialize the language model
llm = ChatOpenAI()

# Define the state structure for the chat
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# Define the chat node function for the graph
def chat_node(state: ChatState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {'messages': [response]}


# Checkpointer to save the state
checkpointer = InMemorySaver()


# Define the state graph
graph = StateGraph(ChatState)
graph.add_node('chat_node', chat_node)
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)


# Compile the graph into a chatbot
chatbot = graph.compile(checkpointer=checkpointer)