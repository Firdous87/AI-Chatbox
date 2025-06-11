import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from dotenv import load_dotenv

from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os
import shutil
from urllib.request import urlopen
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()
GROQ_API_KEY = "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"

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
            page_text = soup.get_text(separator=' ', strip=True)
            text += page_text + "\n\n"
            sources.append(url)
        except Exception as e:
            st.error(f"Error processing URL: {url} - {e}")
            # Do not return immediately here, try to process other URLs
    return text, sources

# Split text
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    return splitter.split_text(text)

# Create/Update vector store using Document objects
def get_vector_store(documents): # Now accepts a list of Document objects
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vector_store = Chroma.from_documents( # Use from_documents instead of from_texts
        documents=documents,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    vector_store.persist()

# Handle user query
def user_input(user_question):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

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
                all_raw_text = "" # Accumulate all raw text here
                source_markers = [] # Store (start_index, end_index, source_url_or_tag)

                # 1. Process PDFs and record their text and markers
                if pdf_docs:
                    pdf_text = get_pdf_text(pdf_docs)
                    if pdf_text: # Only add if text was extracted
                        start_index = len(all_raw_text)
                        all_raw_text += pdf_text + "\n\n" # Add a separator
                        end_index = len(all_raw_text)
                        source_markers.append((start_index, end_index, "PDF Upload"))
                        st.info(f"Processed {len(pdf_docs)} PDF(s).")

                # 2. Process URLs and record their text and markers
                if website_urls:
                    url_text, url_sources_list = get_url_text(website_urls)
                    if url_text: # Only add if text was extracted
                        start_index = len(all_raw_text)
                        all_raw_text += url_text + "\n\n" # Add a separator
                        end_index = len(all_raw_text)
                        # Store specific URLs if available, otherwise a generic tag
                        source_markers.append((start_index, end_index, url_sources_list if url_sources_list else "Website Content"))
                        st.info(f"Processed {len(url_sources_list)} URL(s).")
                    else:
                        st.warning("No text could be extracted from the provided URLs.")


                if not all_raw_text:
                    st.warning("Please upload PDF files or enter website URLs to process.")
                    return

                # 3. Chunk the combined raw text
                combined_chunks = get_text_chunks(all_raw_text)

                # 4. Create Document objects with proper sources
                all_documents = []
                for chunk in combined_chunks:
                    chunk_start = all_raw_text.find(chunk) # Find where this chunk starts in the overall text
                    if chunk_start == -1: # Fallback if direct find fails (due to text splitter nuances)
                         # This happens if chunk text is slightly different after splitting
                         # or if the chunk is not found directly. Fallback to a generic source.
                        all_documents.append(Document(page_content=chunk, metadata={"source": "Mixed Content (Approx.)"}))
                        continue

                    chunk_end = chunk_start + len(chunk)
                    
                    # Determine source for this chunk
                    found_source = "Unknown Source"
                    for start, end, source_info in source_markers:
                        # Check if the chunk overlaps significantly with a source marker
                        if not (chunk_end <= start or chunk_start >= end):
                            if isinstance(source_info, list) and source_info:
                                # For URLs, assign the first URL that this chunk falls into
                                found_source = source_info[0] # Simplification, could be more granular
                                break
                            elif isinstance(source_info, str):
                                found_source = source_info
                                break
                    
                    all_documents.append(Document(page_content=chunk, metadata={"source": found_source}))


                if not all_documents:
                    st.warning("No documents could be prepared for indexing.")
                    return

                get_vector_store(all_documents)
                st.success("Content indexed and ready!")
                
        # Option to clear the database


if __name__ == "__main__":
    main()