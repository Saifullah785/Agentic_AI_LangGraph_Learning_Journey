# pip install -U langchain langchain-openai langchain-community faiss-cpu pypdf python-dotenv langsmith

import os
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

# -------------- helpers (not traced individually) -----------------
@traceable(name='load_pdf')
def load_pdf(path: str):
    loader = PyPDFLoader(path)
    return loader.load()  # list[Document]

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

@traceable(name='setup_pipeline', tags=['setup'])
def setup_pipeline(pdf_path: str, chunk_size=1000, chunk_overlap=150):
    
    docs = load_pdf(pdf_path)
    
    splits = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    vs = build_vectorstore(splits)
    
    return vs

# -------------- model, prompt, and run -----------------

llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

prompt = ChatPromptTemplate.from_template([
    ('system', "Answer only from the provided context. if not found, say you don't know."),
    ('human', "Question: {question}\n\nContext:\n{context}")


])

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

# -----------------------one top-level (root) run -------------------------
@traceable(name='pdf_rag_full_run')
def setup_pipeline_and_query(pdf_path: str, question: str):

    # parent setup run (child of root)
    vectorstore = setup_pipeline(pdf_path, chunk_size=1000, chunk_overlap=150)

    # retrieve relevant docs
    retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 4})

    parallel = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough(),
    })

    chain = parallel | prompt | llm | StrOutputParser()

    # This langChain run stays under the same root (since we are inside this traced function)
    lc_config = {"run_name": "pdf_rag_query"}
    return  chain.invoke(question=question, config=lc_config)

# ----------------------- CLI -------------------------
if __name__ == "__main__":
    print("PDF RAG ready. Ask a question (or ctrl+c to exit).")
    q = input("\nQ: ").strip()
    ans = setup_pipeline_and_query(PDF_PATH, q)
    print(f"\nA: {ans}\n")