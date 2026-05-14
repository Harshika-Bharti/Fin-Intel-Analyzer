from langchain_community.llms import Ollama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

DB_PATH = "./chroma_db"


def save_to_database(documents):
    """
    Saves LangChain Documents into ChromaDB.
    """

    db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=DB_PATH
    )

    return db


def get_financial_answer(user_question):
    """
    Retrieves relevant chunks and generates
    a grounded financial response with citations.
    """

    # Load DB
    db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

    # Better retriever (MMR retrieval)
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 20
        }
    )

    # Local LLM
    llm = Ollama(model="llama3")

    # Stronger hallucination-control prompt
    system_prompt = """
You are an AI financial document analysis assistant.

Use ONLY the provided context to answer the question.

Rules:
1. Do NOT hallucinate.
2. If the answer is not present in the context, say:
   "The uploaded documents do not contain this information."
3. Keep answers factual and concise.
4. Mention page references when possible.

Context:
{context}
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}")
        ]
    )

    # Create QA chain
    question_answer_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    # Create RAG chain
    rag_chain = create_retrieval_chain(
        retriever,
        question_answer_chain
    )

    # Run query
    response = rag_chain.invoke({
        "input": user_question
    })

    # Extract retrieved source docs
    sources = []

    for doc in response["context"]:

        source_info = {
            "page": doc.metadata.get("page"),
            "source": doc.metadata.get("source"),
            "path": doc.metadata.get("path"),
            "content": doc.page_content[:300]
        }

        sources.append(source_info)

    return {
        "answer": response["answer"],
        "sources": sources
    }