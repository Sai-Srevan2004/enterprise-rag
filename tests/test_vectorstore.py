import os
from indexing.vectorstore import load_retriever, CHROMA_DIR

if not os.path.exists(CHROMA_DIR):
    print("ERROR: Index not built yet. Run main.py first.")
    exit()

retriever = load_retriever()

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
    print(f"Query {i}: {query}")
    print('='*60)
    results = retriever.invoke(query)
    print(f"Docs retrieved: {len(results)}")
    for j, doc in enumerate(results, 1):
        print(f"\n--- Result {j} ---")
        print(doc.page_content[:400])