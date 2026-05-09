from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os

# 1. Initialize local embeddings (runs on your M2 chip)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
DB_PATH = "./chroma_db"

def save_to_database(chunks):
    """
    Takes text chunks and saves them into the Vector Database.
    """
    # Create the database and persist it to your Mac
    db = Chroma.from_texts(
        chunks, 
        embeddings, 
        persist_directory=DB_PATH
    )
    return db

def get_financial_answer(user_question):
    """
    Retrieves relevant data and generates an AI answer.
    """
    # Load the existing database
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    
    # Setup the LLM (Ensure your .env has the API key)
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    
    system_prompt = (
        "You are a professional financial assistant. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, "
        "say you don't know. Use three sentences maximum and keep the answer concise.\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # Combine documents and create the retrieval chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    response = rag_chain.invoke({"input": user_question})
    return response["answer"]