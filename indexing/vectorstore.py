
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore
from langchain_classic.storage import create_kv_docstore
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from indexing.chunker import get_splitters

# Project root is 2 levels up from this file
BASE_DIR = Path(__file__).parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"
DOCSTORE_DIR = BASE_DIR / "docstore"


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def build_retriever(docs: list[Document]) -> ParentDocumentRetriever:
    embeddings = get_embeddings()
    parent_splitter, child_splitter = get_splitters()

    vectorstore = Chroma(
        collection_name="child_chunks",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR)
    )

    DOCSTORE_DIR.mkdir(parents=True, exist_ok=True)
    fs = LocalFileStore(str(DOCSTORE_DIR))
    docstore = create_kv_docstore(fs)

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    print("Indexing documents... please wait...")
    retriever.add_documents(docs, add_to_docstore=True)
    print("Done. Index saved to disk.")

    return retriever


def load_retriever() -> ParentDocumentRetriever:
    embeddings = get_embeddings()
    parent_splitter, child_splitter = get_splitters()

    vectorstore = Chroma(
        collection_name="child_chunks",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR)
    )

    fs = LocalFileStore(str(DOCSTORE_DIR))
    docstore = create_kv_docstore(fs)

    return ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )