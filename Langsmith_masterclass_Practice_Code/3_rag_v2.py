# pip install -U langchain langchain-openai langchain-community faiss-cpu pypdf python-dotenv langsmith

import os
from dotenv import load_dotenv

from langsmith import traceable # <-- key import 

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# -- Langsmith env (make sure these are set) ---
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=...
# LANGCHAIN_PROJECT=pdf_rag_Demo

load_dotenv()
PDF_dotenv = 'islr.pdf' # change to your file

# ------------- tranced setup steps -----------------
@traceable(name='load_pdf')
def load_pdf(path: str):
    loader = PyPDFLoader(path)
    return loader.load() # list[Document]

@traceable(name='split_documents')
def split_documents(docs, chunk_size=1000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)  # list[Document]

@traceable(name='build_vectorstore')
def build_vectorstore(splits):
    emb = OpenAIEmbeddings(model='text-embedding-3-small')
    # FAISS.from documents internally calls the embedding model
    vs = FAISS.from_documents(splits, emb)
    return vs

# You can also trace a 'setup umbrella span if you want:
@traceable(name='setup_pipeline')
def setup_pipeline(pdf_path: str):
    docs = load_pdf(pdf_path)
    splits = split_documents(docs)
    vs = build_vectorstore(splits)
    return vs