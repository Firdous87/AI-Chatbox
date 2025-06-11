import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document # This import is now directly used
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
                all_documents = [] # This will store Langchain Document objects

                if pdf_docs:
                    pdf_text = get_pdf_text(pdf_docs)
                    pdf_chunks = get_text_chunks(pdf_text)
                    for i, chunk in enumerate(pdf_chunks):
                        # For PDFs, you can use a generic "PDF Upload" or assign filenames if you tracked them
                        all_documents.append(Document(page_content=chunk)) 
                        # Or if you want to be more specific with PDF names:
                        # all_documents.append(Document(page_content=chunk, metadata={"source": pdf_docs[0].name if pdf_docs else "PDF Upload"}))


                if website_urls:
                    # Collect all URL text and their original URLs
                    full_url_text = ""
                    distinct_urls_processed = [] # To ensure unique URLs are in sources
                    for url in website_urls:
                        current_url_text, current_url_source_list = get_url_text([url])
                        if current_url_text: # Only add if text was successfully extracted
                            full_url_text += current_url_text
                            distinct_urls_processed.extend(current_url_source_list)

                    if full_url_text: # Only chunk if there's actual text from URLs
                        url_chunks = get_text_chunks(full_url_text)
                        
                        # Distribute URL sources to chunks. This is still a simplification.
                        # A better method would be to pass individual Document objects
                        # for each URL, each with its own URL as metadata, then chunk them.
                        # But for combined raw text, we distribute sources.
                        source_idx = 0
                        for chunk in url_chunks:
                            source = "Website Content"
                            if distinct_urls_processed:
                                source = distinct_urls_processed[source_idx % len(distinct_urls_processed)]
                            all_documents.append(Document(page_content=chunk, metadata={"source": source}))
                            source_idx += 1


                if not all_documents:
                    st.warning("Please upload PDF files or enter website URLs to process.")
                    return

                get_vector_store(all_documents) # Pass the list of Document objects
                print(all_documents)
                st.success("Content indexed and ready!")
                
        # Option to clear the database


if __name__ == "__main__":
    main()