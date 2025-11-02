# python
# filepath: /Users/gauravsingh/study/AI/DeependraBhaiyaproject/questionier/questionnaires/ingest_vectors.py
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Optional

# Loaders / splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Resilient imports for loaders
CSVLoader = None
DirectoryLoader = None
PyPDFLoader = None
TextLoader = None
try:
    from langchain_community.document_loaders import CSVLoader as _CSVLoader
    from langchain_community.document_loaders import DirectoryLoader as _DirectoryLoader
    from langchain_community.document_loaders import PyPDFLoader as _PyPDFLoader
    from langchain_community.document_loaders import TextLoader as _TextLoader
    CSVLoader, DirectoryLoader, PyPDFLoader, TextLoader = _CSVLoader, _DirectoryLoader, _PyPDFLoader, _TextLoader
except Exception:
    pass

# Resilient imports for vectorstore + embeddings
Chroma = None
HuggingFaceEmbeddings = None
try:
    from langchain_chroma import Chroma
except Exception:
    try:
        from langchain_community.vectorstores import Chroma
    except Exception:
        Chroma = None

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except Exception:
        HuggingFaceEmbeddings = None


def resolve_path(path_str: Optional[str]) -> Optional[Path]:
    if not path_str:
        return None
    p = Path(path_str).expanduser()
    if p.exists():
        return p.resolve()
    # try relative to project data folder
    proj_root = Path(__file__).resolve().parent
    candidate = (proj_root / path_str).resolve()
    if candidate.exists():
        return candidate
    # if only a filename, try ./data/<filename>
    if not p.parent or str(p.parent) == ".":
        data_candidate = (proj_root / "data" / p.name).resolve()
        if data_candidate.exists():
            return data_candidate
    return None


def split_docs(docs: List, chunk_size=800, chunk_overlap=120):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)


def load_csv(file_path: Path) -> List:
    if CSVLoader is None:
        # Fallback: read whole file as one document
        text = file_path.read_text(errors="ignore")
        from langchain_core.documents import Document
        return [Document(page_content=text, metadata={"source": str(file_path)})]
    return CSVLoader(file_path=str(file_path), csv_args={"delimiter": ","}).load()


def load_pdf_dir(dir_path: Path) -> List:
    if DirectoryLoader is None or PyPDFLoader is None:
        return []
    return DirectoryLoader(str(dir_path), glob="**/*.pdf", loader_cls=PyPDFLoader, use_multithreading=True).load()


def load_text_dir(dir_path: Path) -> List:
    if DirectoryLoader is None or TextLoader is None:
        return []
    return DirectoryLoader(str(dir_path), glob="**/*.txt", loader_cls=TextLoader, use_multithreading=True).load()


def ingest(collection: Optional[str], csv: Optional[str], pdf_dir: Optional[str], text_dir: Optional[str], persist_dir: str):
    if Chroma is None or HuggingFaceEmbeddings is None:
        raise ImportError("Install chroma + embeddings: langchain-chroma and langchain-huggingface (or community equivalents).")

    docs: List = []

    csv_p = resolve_path(csv) if csv else None
    pdf_p = resolve_path(pdf_dir) if pdf_dir else None
    txt_p = resolve_path(text_dir) if text_dir else None

    if csv_p:
        print(f"Loading CSV: {csv_p}")
        docs += load_csv(csv_p)
    if pdf_p:
        print(f"Loading PDFs from: {pdf_p}")
        docs += load_pdf_dir(pdf_p)
    if txt_p:
        print(f"Loading TXTs from: {txt_p}")
        docs += load_text_dir(txt_p)

    if not docs:
        raise FileNotFoundError("No documents found. Pass valid paths (hint: use ./data/your.csv or absolute paths).")

    print(f"Loaded {len(docs)} documents. Splitting into chunks...")
    chunks = split_docs(docs)

    print(f"Embedding and persisting to Chroma at {persist_dir} ...")
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if collection:
        vs = Chroma.from_documents(documents=chunks, embedding=emb, persist_directory=persist_dir, collection_name=collection)
    else:
        vs = Chroma.from_documents(documents=chunks, embedding=emb, persist_directory=persist_dir)
    try:
        vs.persist()
    except Exception:
        pass
    print(f"Ingested {len(chunks)} chunks into {persist_dir}{' (collection='+collection+')' if collection else ''}")


def main():
    ap = argparse.ArgumentParser(description="Ingest CSV/PDF/TXT into Chroma vectors DB.")
    ap.add_argument("--collection", help="Collection name (optional).")
    ap.add_argument("--csv", help="Path to a CSV file to ingest.")
    ap.add_argument("--pdf-dir", help="Directory of PDFs to ingest.")
    ap.add_argument("--text-dir", help="Directory of TXT files to ingest.")
    ap.add_argument("--persist", default="vectors", help="Persist directory (default: ./vectors).")
    args = ap.parse_args()

    ingest(args.collection, args.csv, args.pdf_dir, args.text_dir, args.persist)


if __name__ == "__main__":
    main()