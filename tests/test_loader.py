import os
from indexing.loader import load_and_prepare

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

docs = load_and_prepare(DATA_DIR)

print(f"\nTotal docs: {len(docs)}")
print(f"\nFirst doc preview:")
print(docs[0].page_content[:300])
print(f"\nMetadata: {docs[0].metadata}")