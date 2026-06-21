from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

PROMPT = ChatPromptTemplate.from_template("""
You are an enterprise policy assistant for Rikalp Capital.
Answer the question based ONLY on the context below.
If the answer is not in the context, say exactly:
"I don't have enough information in the provided documents to answer this."

Context:
{context}

Question: {question}

Instructions for your answer:
- Answer in 1-3 clear sentences
- End EVERY answer with this exact format on a new line:
  Source: [filename] | Page: [page number]
- Never put the source anywhere else
- Never skip the source line

Answer:
""")


def format_docs(docs):
    return "\n\n---\n\n".join(
        f"[Source: {d.metadata.get('filename', 'unknown')} | "
        f"Page: {d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in docs
    )


def build_basic_chain(retriever):
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return chain