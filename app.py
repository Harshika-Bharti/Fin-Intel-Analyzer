import streamlit as st
from processor import process_pdf  # This imports your new file!

st.set_page_config(page_title="Fin-Intel Pro", page_icon="📈")

st.title("Financial Intelligence Engine 📈")
st.markdown("---")

uploaded_file = st.file_uploader("Upload a Financial PDF (Annual Report, 10-K, etc.)", type="pdf")

if uploaded_file:
    # This 'spinner' makes the UI look professional while the Mac processes the file
    with st.spinner("Ripping through the PDF..."):
        # We call the function from processor.py
        data_chunks = process_pdf(uploaded_file)
        
        st.success(f"Successfully processed! Found {len(data_chunks)} sections of data.")
        
        # Show the first chunk so we can verify it's working
        with st.expander("View Raw Data Preview"):
            st.write(data_chunks[0])
