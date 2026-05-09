from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

def process_pdf(pdf_file):
    """
    This function takes an uploaded PDF, extracts the text, 
    and breaks it into manageable chunks for the AI.
    """
    # 1. Initialize the PDF Reader
    reader = PdfReader(pdf_file)
    full_text = ""
    
    # 2. Loop through every page and grab the text
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

    # 3. Define the 'Chunking' strategy
    # Financial data needs context, so we overlap the chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,    # Each snippet is ~2 paragraphs
        chunk_overlap=200,  # 200 characters from Chunk 1 repeat in Chunk 2
        length_function=len,
        is_separator_regex=False,
    )
    
    # 4. Create the chunks
    chunks = text_splitter.split_text(full_text)
    return chunks