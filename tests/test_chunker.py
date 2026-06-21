# tests/test_chunker.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from indexing.loader import load_and_prepare
from indexing.chunker import get_splitters, analyze_chunks

DATA_DIR = Path(__file__).parent.parent / "data"

# Load docs first
docs = load_and_prepare(str(DATA_DIR))

# Get splitters
parent_splitter, child_splitter = get_splitters()

# Split into parent chunks
parent_chunks = parent_splitter.split_documents(docs)
analyze_chunks(parent_chunks, "Parent Chunks")

# Split into child chunks
child_chunks = child_splitter.split_documents(docs)
analyze_chunks(child_chunks, "Child Chunks")

# Manually inspect a few chunks
print("\n=== Random Parent Chunks ===")
for i in [0, 5, 10]:
    if i < len(parent_chunks):
        print(f"\n--- Parent Chunk {i} ---")
        print(parent_chunks[i].page_content)
        print(f"Metadata: {parent_chunks[i].metadata}")

print("\n=== Random Child Chunks ===")
for i in [0, 5, 10]:
    if i < len(child_chunks):
        print(f"\n--- Child Chunk {i} ---")
        print(child_chunks[i].page_content)
        print(f"Metadata: {child_chunks[i].metadata}")