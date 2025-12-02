# pip install -U langchain langchain-openai langchain-community faiss-cpu pypdf python-dotenv langsmith

import os
import json
import hashlib
from pathlib import Path
from dotenv import load_dotenv

from langsmith import traceable

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

PDF_PATH = 'islr.pdf'  # change to your file
INDEX_ROOT = Path('./indexes')
INDEX_ROOT.mkdir(exist_ok=True)


# -------------- helpers (traced ) -----------------

@traceable(name="load_pdf")

def load_pdf(path: str):

    return PyPDFLoader(path).load()

@traceable(name="split_documents")

def split_documents(docs, chunk_size=1000, chunk_overlap=150):

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

@traceable(name='build_vectorstore')

def build_vectorstore(split, embed_model_name, index_path: str):

    emb = OpenAIEmbeddings(model=embed_model_name)

    return FAISS.from_documents(split, emb)