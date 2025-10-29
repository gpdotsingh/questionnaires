# ingest_vectors.py
import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Choose embeddings via env var:
#   EMBEDDINGS=sentence (default) | ollama | openai
USE = os.getenv("EMBEDDINGS", "sentence")

if USE == "sentence":
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
elif USE == "ollama":
    # Requires: `ollama serve` and a pulled embed model (e.g. nomic-embed-text)
    from langchain_ollama import OllamaEmbeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
elif USE == "openai":
    from langchain_openai import OpenAIEmbeddings
    # Requires OPENAI_API_KEY
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
else:
    raise ValueError(f"Unknown EMBEDDINGS={USE}")

DATA_DIR = Path("data")
PERSIST_DIR = Path("vectors")  # Chroma will store the DB here

def load_docs():
    docs = []
    if DATA_DIR.exists():
        docs += DirectoryLoader(str(DATA_DIR), glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True).load()
        docs += DirectoryLoader(str(DATA_DIR), glob="**/*.txt", loader_cls=TextLoader, show_progress=True).load()
        docs += DirectoryLoader(str(DATA_DIR), glob="**/*.md",  loader_cls=TextLoader, show_progress=True).load()
    return docs

def main():
    print(f"Embeddings backend: {USE}")
    docs = load_docs()
    if not docs:
        print("No docs found in ./data (pdf/txt/md). Add files and re-run.")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=120, separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    print(f"Loaded {len(docs)} docs → {len(chunks)} chunks")

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(PERSIST_DIR),
        collection_name="qa_docs",
    )
    vectordb.persist()
    print(f"Chroma DB persisted at: {PERSIST_DIR.resolve()}")

if __name__ == "__main__":
    main()
