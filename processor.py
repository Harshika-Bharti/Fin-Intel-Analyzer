from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pypdf import PdfReader


def process_pdf(pdf_file, file_path):
    """
    Processes an uploaded PDF and converts it into
    metadata-rich LangChain Documents.

    Each chunk now contains:
    - page content
    - page number
    - source file name

    This enables:
    - citations
    - source tracking
    - better retrieval
    - multi-document support
    """

    # Load PDF
    reader = PdfReader(pdf_file)

    # Improved chunking strategy
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    documents = []

    # Process each page separately
    for page_number, page in enumerate(reader.pages, start=1):

        # Extract page text
        text = page.extract_text()

        # Skip empty pages
        if not text:
            continue

        # Split current page into chunks
        chunks = text_splitter.split_text(text)

        # Convert chunks into LangChain Documents
        for chunk in chunks:

            doc = Document(
                page_content=chunk,
                metadata={
                    "page": page_number,
                    "source": pdf_file.name,
                    "path": file_path
                }
            )

            documents.append(doc)

    return documents
