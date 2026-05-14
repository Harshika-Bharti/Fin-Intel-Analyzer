import streamlit as st
import os
from processor import process_pdf
from engine import save_to_database, get_financial_answer

st.title("Financial Intelligence Engine 📈")

uploaded_file = st.file_uploader(
    "Upload Report",
    type="pdf"
)

if uploaded_file:

    # Create upload directory
    UPLOAD_DIR = "uploaded_reports"
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # Define file path
    file_path = os.path.join(
        UPLOAD_DIR,
        uploaded_file.name
    )

    # Save uploaded PDF locally
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Process only once per session
    if "processed" not in st.session_state:

        with st.spinner("Analyzing Financial Data..."):

            chunks = process_pdf(
                uploaded_file,
                file_path
            )

            save_to_database(chunks)

            st.session_state.processed = True

            st.success("Analysis Complete!")

    # Chat UI
    user_input = st.chat_input(
        "Ask about revenue, risks, or trends..."
    )

    if user_input:

        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                response = get_financial_answer(
                    user_input
                )

                st.write(response["answer"])

                # Show citations
                with st.expander("View Sources"):

                    for source in response["sources"]:

                        st.markdown(
                            f"### 📄 Page {source['page']} | {source['source']}"
                        )

                        with open(source["path"], "rb") as pdf_file:

                            st.download_button(
                                label="Open Source PDF",
                                data=pdf_file,
                                file_name=source["source"],
                                mime="application/pdf",
                                key=f"{source['source']}_{source['page']}"
                            )

                        st.write(source["content"])