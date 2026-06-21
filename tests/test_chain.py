from pathlib import Path
from dotenv import load_dotenv
from generation.basic_chain import build_basic_chain
from indexing.vectorstore import build_retriever, load_retriever, CHROMA_DIR
from pathlib import Path
from indexing.loader import load_and_prepare

load_dotenv()

BASE_DIR = Path(__file__).parent.parent   # goes up from tests/ to project root
DATA_DIR = BASE_DIR / "data"              # project root / data


if not CHROMA_DIR.exists():
    print("First time — building index...")
    docs = load_and_prepare(str(DATA_DIR))
    retriever = build_retriever(docs)
else:
    print("Index found — loading...")
    retriever = load_retriever()

retriever = load_retriever()
chain = build_basic_chain(retriever)

questions = [
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

for q in questions:
    print(f"\nQ: {q}")
    answer = chain.invoke(q)
    print(f"A: {answer}")
    print("-" * 60)