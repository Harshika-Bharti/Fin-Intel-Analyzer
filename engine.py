from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
# CHANGE THIS:
# from langchain.chains import create_retrieval_chain 
# TO THIS:
from langchain_classic.chains import create_retrieval_chain

# AND THIS:
# from langchain.chains.combine_documents import create_stuff_documents_chain
# TO THIS:
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Initialize local embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_financial_answer(user_question):
    # 1. Load the database
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    
    # 2. Setup the Brain (LLM)
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    
    # 3. Create a specialized Financial Prompt
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

    # 4. Create the modern Chains
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    # 5. Get the result using 'invoke' (the new standard instead of 'run')
    response = rag_chain.invoke({"input": user_question})
    return response["answer"]