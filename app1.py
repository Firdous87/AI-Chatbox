import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document # Keep this import, though not directly used for text_chunks
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os
import shutil # Still useful if you want to clear chroma_db
from urllib.request import urlopen
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Set up Groq model (Llama3-70b-8192)
def get_conversational_chain():
    prompt_template = """
Answer the question as detailed as possible from the provided context.
If the answer is not in the context, just say "answer is not available in the context".

Context:
{context}

Question:
{question}

Answer:
"""
    llm = ChatGroq(model_name="llama3-70b-8192", groq_api_key=GROQ_API_KEY)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(llm, chain_type="stuff", prompt=prompt)

# Extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text

# Extract text from URLs
def get_url_text(urls):
    text = ""
    sources = []
    for url in urls:
        try:
            html = urlopen(url).read()
            soup = BeautifulSoup(html, "html.parser")
            # Get text, preserving some structure with spaces
            page_text = soup.get_text(separator=' ', strip=True)
            text += page_text + "\n\n"
            sources.append(url)
        except Exception as e:
            st.error(f"Error processing URL: {url} - {e}")
            # Do not return immediately here, try to process other URLs
    return text, sources

# Split text
def get_text_chunks(text):
    # Using a smaller chunk size and overlap for more granular context, adjust as needed
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    return splitter.split_text(text)

# Create/Update vector store with sources
def get_vector_store(text_chunks, sources):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create Document objects with content and source metadata
    documents = []
    # If sources is a list of URLs, map them to chunks. This is a simplification.
    # A more robust solution would track which chunk came from which specific URL/PDF.
    # For now, we'll assign sources in a rotating fashion if more chunks than sources,
    # or use "PDF Upload" for PDF chunks.
    
    # This logic assumes 'sources' list is pre-aligned with chunks or uses generic tags.
    # Let's ensure text_chunks and sources are correctly matched.
    
    # The 'sources' list coming into this function needs to correspond to the 'text_chunks'.
    # A simpler approach for the get_vector_store is to create documents with metadata
    # for each chunk. The main loop will handle creating a combined text and sources.
    
    # We'll adjust how `text_chunks` and `sources` are handled in the main function
    # to better reflect individual chunk origins.
    
    # For the `get_vector_store` function itself, assume `sources` is a list that aligns
    # with `text_chunks` (e.g., each source is either a URL or "PDF Upload").
    
    # Re-writing metadatas creation for clarity:
    metadatas = []
    for i, chunk in enumerate(text_chunks):
        if i < len(sources):
            metadatas.append({"source": sources[i]})
        else:
            # Fallback if text_chunks count exceeds explicit sources (e.g., from generic "PDF Upload")
            metadatas.append({"source": "Processed Document"}) 

    vector_store = Chroma.from_texts(
        texts=text_chunks,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory="chroma_db"
    )
    vector_store.persist()

# Handle user query
def user_input(user_question):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load the persisted vector store
    try:
        db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    except Exception as e:
        st.error(f"Error loading vector store. Please process documents and URLs first: {e}")
        return

    docs = db.similarity_search(user_question)
    
    if not docs:
        st.write("Reply: I could not find relevant information in the processed documents to answer your question.")
        return

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply:", response["output_text"])

    st.subheader("Sources:")
    # Display unique sources from the retrieved documents
    unique_sources = set()
    for doc in docs:
        source = doc.metadata.get("source")
        if source:
            unique_sources.add(source)
    
    if unique_sources:
        for src in unique_sources:
            st.markdown(f"- [{src}]({src})")
    else:
        st.write("- No specific sources found for the retrieved content.")


# Streamlit UI
def main():
    st.set_page_config(page_title="Chat with Documents (Groq)")
    st.header("Chat with multiple PDFs and Websites using Llama3 (Groq) 💬")

    user_question = st.text_input("Ask a question from the processed files and websites:")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Upload & Process Content:")
        
        pdf_docs = st.file_uploader("Upload PDF files", accept_multiple_files=True)
        website_urls_str = st.text_area("Enter website URLs (one per line):")
        website_urls = [url.strip() for url in website_urls_str.split('\n') if url.strip()]

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                all_text = ""
                all_chunks_with_sources = [] # To store (chunk_text, source) tuples

                if pdf_docs:
                    pdf_text = get_pdf_text(pdf_docs)
                    pdf_chunks = get_text_chunks(pdf_text)
                    for chunk in pdf_chunks:
                        all_chunks_with_sources.append((chunk, "PDF Upload"))
                    all_text += pdf_text # Accumulate text for potential future combined use if needed

                if website_urls:
                    url_text, url_sources = get_url_text(website_urls)
                    url_chunks = get_text_chunks(url_text)
                    
                    # Associate each URL chunk with its source URL
                    # This is a basic way; a more robust way might require tracking the exact URL for each chunk
                    # For simplicity, we'll cycle through the provided URLs for URL chunks
                    if url_sources:
                        source_index = 0
                        for chunk in url_chunks:
                            all_chunks_with_sources.append((chunk, url_sources[source_index % len(url_sources)]))
                            source_index += 1
                    else:
                        # Fallback if URLs failed to process
                        for chunk in url_chunks:
                            all_chunks_with_sources.append((chunk, "Website Content"))
                    all_text += url_text # Accumulate text
                    print(all_text)

                if not all_chunks_with_sources:
                    st.warning("Please upload PDF files or enter website URLs to process.")
                    return

                # Separate chunks and sources for get_vector_store
                combined_chunks = [item[0] for item in all_chunks_with_sources]
                combined_sources = [item[1] for item in all_chunks_with_sources]

                get_vector_store(combined_chunks, combined_sources)
                st.success("Content indexed and ready!")
                
        # Option to clear the database
        if st.button("Clear Indexed Data"):
            if os.path.exists("chroma_db"):
                shutil.rmtree("chroma_db")
                st.success("Indexed data cleared!")
            else:
                st.info("No indexed data to clear.")

if __name__ == "__main__":
    main()