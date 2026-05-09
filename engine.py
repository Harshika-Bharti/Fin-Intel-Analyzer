from langchain_community.llms import Ollama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os

# 1. Initialize local embeddings (Still using HuggingFace on your M2)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
DB_PATH = "./chroma_db"

def save_to_database(chunks):
    """
    Takes text chunks and saves them into the local Chroma Vector Database.
    """
    db = Chroma.from_texts(
        chunks, 
        embeddings, 
        persist_directory=DB_PATH
    )
    return db

def get_financial_answer(user_question):
    """
    Retrieves data from Chroma and generates an answer using local Llama 3.
    """
    # Load the existing database from your Mac
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    
    # 2. Setup the Local LLM (Ollama)
    # This uses the Llama 3 model you just downloaded!
    llm = Ollama(model="llama3")
    
    system_prompt = (
        "You are a professional financial assistant. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, "
        "say you don't know. Use three sentences maximum and keep the answer concise.\n\n"
        "Context: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # Create the modern RAG chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    # Get the response locally
    response = rag_chain.invoke({"input": user_question})
    return response["answer"]