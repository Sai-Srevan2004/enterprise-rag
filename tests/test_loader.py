from pathlib import Path
from indexing.loader import load_and_prepare

DATA_DIR = Path(__file__).parent.parent / "data"

docs = load_and_prepare(str(DATA_DIR))

print(f"\nTotal docs: {len(docs)}")
print(f"\nFirst doc preview:")
print(docs[0].page_content[:300])
print(f"\nMetadata: {docs[0].metadata}")