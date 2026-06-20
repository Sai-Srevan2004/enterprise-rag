import re
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_core.documents import Document


def load_documents(source_path: str) -> list[Document]:
    path = Path(source_path)

    if path.is_dir():
        loader = DirectoryLoader(
            str(path),
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
            silent_errors=True
        )
    else:
        loader = PyPDFLoader(str(path))

    docs = loader.load()
    print(f"Loaded {len(docs)} pages from {source_path}")
    return docs


def clean_document(doc: Document) -> Document:
    text = doc.page_content

    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
    text = text.strip()

    return Document(page_content=text, metadata=doc.metadata)


def enrich_metadata(doc: Document, category: str = "policy") -> Document:
    doc.metadata["category"] = category
    doc.metadata["word_count"] = len(doc.page_content.split())
    doc.metadata["filename"] = Path(doc.metadata.get("source", "")).name
    return doc


def load_and_prepare(source_path: str, category: str = "policy") -> list[Document]:
    raw_docs = load_documents(source_path)

    cleaned = []
    for doc in raw_docs:
        cleaned_doc = clean_document(doc)
        if len(cleaned_doc.page_content) < 100:
            continue
        enriched = enrich_metadata(cleaned_doc, category)
        cleaned.append(enriched)

    print(f"After cleaning: {len(cleaned)} usable pages")
    return cleaned