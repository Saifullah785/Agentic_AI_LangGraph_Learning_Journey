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

# ----------------- cache key / fingerprint -----------------

def _file_fingerprint(path: str):

    p = Path(path)
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return {"sha256": h.hexdigest(), "size": p.stat().st_size, "mtime": int(p.stat().st_mtime)}

def _index_key(pdf_path: str, chunk_size: int, chunk_overlap: int, embed_model_name: str) -> str:
    meta = {
        "pdf_fingerprint": _file_fingerprint(pdf_path),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": embed_model_name,
        "format": "v1",
    }
    return hashlib.sha256(json.dumps(meta, sort_keys=True).encode("utf-8")).hexdigest()

# ----------------- explicitly traced load/build runs -----------------

@traceable(name='load_index', tags={'index'})
def load_index_run(index_dir, Path, embed_model_name: str):
    emb = OpenAIEmbeddings(model=embed_model_name)
    return FAISS.load_local(
        str(index_dir),
        emb,
        allow_dangerous_deserialization=True
    )


@traceable(name='build_index', tags={'index'})
def build_index_run(pdf_path: str, chunk_size: int, chunk_overlap: int, embed_model_name: str, index_dir: Path):
    docs = load_pdf(pdf_path)
    splits = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    vs = build_vectorstore(splits, embed_model_name)
    index_dir.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(index_dir))
    (index_dir / 'meta.json').write_text(json.dumps({
        "pdf_path": os.path.abspath(pdf_path),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": embed_model_name,
        "format": "v1",
    }, indent=2))
    return vs

# ----------------- dispatcher (not traced) -----------------

def load_or_build_index(
    pdf_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    embed_model_name: str = "text-embedding-3-small",
    force_rebuild: bool = False,
):
    key = _index_key(pdf_path, chunk_size, chunk_overlap, embed_model_name)
    index_dir = INDEX_ROOT / key
    cache_hit = index_dir.exists() and not force_rebuild
    if cache_hit:
        return load_index_run(index_dir, embed_model_name)
    else:
        return build_index_run(pdf_path, chunk_size, chunk_overlap, embed_model_name, index_dir)
    

# ----------------- model, prompt and pipeline -----------------









