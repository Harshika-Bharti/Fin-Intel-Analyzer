from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Using the local embeddings we discussed
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_financial_answer(user_question):
    # 1. Load the database from your Mac
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    
    # 2. Setup the Brain (The LLM)
    # If using OpenAI:
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0) 
    
    # 3. Create the 'Chain' (The link between Search and Answer)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # 'Stuff' just means 'stuff these paragraphs into the prompt'
        retriever=db.as_retriever(search_kwargs={"k": 3})
    )
    
    # 4. Get the result
    response = qa_chain.invoke(user_question)
    return response["result"]