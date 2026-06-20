from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def get_splitters():
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=30,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    return parent_splitter, child_splitter


def analyze_chunks(chunks: list[Document], label: str = "chunks"):
    sizes = [len(c.page_content) for c in chunks]
    print(f"\n--- {label} analysis ---")
    print(f"Count : {len(chunks)}")
    print(f"Min   : {min(sizes)} chars")
    print(f"Max   : {max(sizes)} chars")
    print(f"Mean  : {int(sum(sizes)/len(sizes))} chars")

    tiny = [c for c in chunks if len(c.page_content) < 50]
    print(f"Tiny (<50 chars): {len(tiny)}")

    sample = chunks[5] if len(chunks) > 5 else chunks[0]
    print(f"\nSample chunk:")
    print(sample.page_content)