from pathlib import Path
from dotenv import load_dotenv
from indexing.vectorstore import load_retriever, CHROMA_DIR
from generation.basic_chain import build_basic_chain

load_dotenv()

if not CHROMA_DIR.exists():
    print("ERROR: Index not built yet. Run main.py first.")
    exit()

retriever = load_retriever()
chain = build_basic_chain(retriever)

questions = [
    "How many earned leaves do employees get per year?",
    "What is the maternity leave policy?",
    "What are the office working hours?",
    "What is the dress code on Saturdays?",
    "How many casual leaves are allowed at once?",
    "How do I report unethical behavior in the company?",
]

for q in questions:
    print(f"\nQ: {q}")
    answer = chain.invoke(q)
    print(f"A: {answer}")
    print("-" * 60)