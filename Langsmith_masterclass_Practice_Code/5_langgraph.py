# pip install -u langgraph langchain-openai pydantic python-dotenv langsmith


import operator
from typing import TypedDict, Annotated, List

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langsmith import traceable
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

# --------------- Setup ---------------------------

load_dotenv()
model = ChatOpenAI(model="gpt-4-mini", temperature=0)

# ---------- Structured schema & model ----------

class EvaluationSchema(BaseModel):
    feedback: str = Field(description="Detailed feedback for the essay")
    score: int = Field(description="Score out of 10", ge=0, le=10)

structured_model = model.with_structured_output(EvaluationSchema)