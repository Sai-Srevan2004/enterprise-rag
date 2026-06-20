import os
from dotenv import load_dotenv
from indexing.loader import load_and_prepare
from indexing.vectorstore import build_retriever, load_retriever, CHROMA_DIR
from generation.basic_chain import build_basic_chain

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# ── Build or Load Index ──────────────────────────────────────
if not os.path.exists(CHROMA_DIR):
    print("First time — building index...")
    docs = load_and_prepare(DATA_DIR)
    retriever = build_retriever(docs)
else:
    print("Index found — loading...")
    retriever = load_retriever()

# ── Build Chain ──────────────────────────────────────────────
chain = build_basic_chain(retriever)

# ── Run Queries ──────────────────────────────────────────────
test_queries = [
    "How many earned leaves do employees get per year?",
    "Can I take leave during my notice period?",
    "What is the maternity leave policy?",
    "What are the office working hours?",
    "What happens if an employee is absent for more than 5 days without informing?",
    "How many casual leaves are allowed at once?",
    "What is the dress code on Saturdays?",
    "How do I report unethical behavior in the company?",
    "How many leaves can be carried forward to next year?",
    "What are the attendance rules for late arrivals?",
]

for i, query in enumerate(test_queries, 1):
    print(f"\n{'='*60}")
    print(f"Q{i}: {query}")
    print('='*60)
    answer = chain.invoke(query)
    print(f"Answer: {answer}")