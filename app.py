import streamlit as st
from processor import process_pdf
from engine import save_to_database, get_financial_answer

st.title("Financial Intelligence Engine 📈")

uploaded_file = st.file_uploader("Upload Report", type="pdf")

if uploaded_file:
    # We only process if it hasn't been done yet for this session
    if "processed" not in st.session_state:
        with st.spinner("Analyzing Financial Data..."):
            chunks = process_pdf(uploaded_file)
            save_to_database(chunks)
            st.session_state.processed = True
            st.success("Analysis Complete!")

    # Chat UI
    user_input = st.chat_input("Ask about revenue, risks, or trends...")
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
            
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = get_financial_answer(user_input)
                st.write(answer)