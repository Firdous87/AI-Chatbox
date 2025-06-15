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
import numpy as np # Import numpy
from sklearn.metrics.pairwise import cosine_similarity # Import cosine_similarity

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
def get_pdf_texts_with_sources(pdf_docs):
    all_pdf_data = []
    for pdf in pdf_docs:
        text = ""
        reader = PdfReader(pdf)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
        if text:
            all_pdf_data.append((text, pdf.name))
    return all_pdf_data

# Extract text from URLs
def get_url_text(urls):
    all_url_text = []
    processed_sources = []
    for url in urls:
        try:
            html = urlopen(url).read()
            soup = BeautifulSoup(html, "html.parser")
            page_text = soup.get_text(separator=' ', strip=True)
            all_url_text.append((page_text, url))
            processed_sources.append(url)
        except Exception as e:
            st.error(f"Error processing URL: {url} - {e}")
    return all_url_text, processed_sources

# Split text
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    return splitter.split_text(text)

# Create/Update vector store with sources
# This function now correctly takes texts and metadatas as arguments
def get_vector_store(texts, metadatas): # Corrected: texts and metadatas are now arguments
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists("chroma_db"):
        shutil.rmtree("chroma_db")

    vector_store = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
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

    docs = db.similarity_search(user_question, k=10) # Increased k to retrieve more potential context

    print("\n--- Retrieved Documents and Their Sources ---")
    if docs:
       for i, doc in enumerate(docs):
        print(f"Document {i+1} (Source: {doc.metadata.get('source', 'N/A')}):")
        print(f"  Content: {doc.page_content[:200]}...")
        print("-" * 30)
    else:
        print("No documents retrieved.")
    print("-------------------------------------------\n")

    if not docs:
        st.write("Reply: I could not find relevant information in the processed documents to answer your question.")
        return

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply:", response["output_text"])

    st.subheader("Sources (likely used):")
    unique_sources_from_answer = set()
    answer_text = response["output_text"]

    answer_embedding = embeddings.embed_query(answer_text)

    for doc in docs:
        source = doc.metadata.get("source")
        if source:
            doc_content = doc.page_content
            doc_embedding = embeddings.embed_query(doc_content)

            similarity = cosine_similarity(
                np.array(answer_embedding).reshape(1, -1),
                np.array(doc_embedding).reshape(1, -1)
            )[0][0]

            if similarity > 0.7: # Tunable threshold
                unique_sources_from_answer.add(source)

    if unique_sources_from_answer:
        for src in unique_sources_from_answer:
            st.markdown(f"- [{src}]({src})")
    else:
        st.write("- No highly relevant sources found in the answer. This could mean the answer was heavily summarized or inferred from multiple sources.")
        st.subheader("All retrieved context sources (for reference):")
        all_retrieved_sources = set()
        for doc in docs:
            source = doc.metadata.get("source")
            if source:
                all_retrieved_sources.add(source)
        for src in all_retrieved_sources:
            st.markdown(f"- [{src}]({src})")


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
                all_chunk_source_pairs = []

                if pdf_docs:
                    pdf_data_with_sources = get_pdf_texts_with_sources(pdf_docs)
                    for pdf_text, pdf_name in pdf_data_with_sources:
                        pdf_chunks = get_text_chunks(pdf_text)
                        for chunk in pdf_chunks:
                            all_chunk_source_pairs.append((chunk, pdf_name))

                if website_urls:
                    url_text_with_sources, _ = get_url_text(website_urls)
                    for page_text, source_url in url_text_with_sources:
                        url_chunks = get_text_chunks(page_text)
                        for chunk in url_chunks:
                            all_chunk_source_pairs.append((chunk, source_url))

                if not all_chunk_source_pairs:
                    st.warning("Please upload PDF files or enter website URLs to process.")
                    return

                # Ensure the chroma_db directory is clean before processing new content
                if os.path.exists("chroma_db"):
                    shutil.rmtree("chroma_db")

                # Define texts and metadatas before passing them to get_vector_store
                texts = [pair[0] for pair in all_chunk_source_pairs]
                metadatas = [{"source": pair[1]} for pair in all_chunk_source_pairs]

                # Now call get_vector_store with the correctly defined variables
                get_vector_store(texts, metadatas) # Corrected: passing texts and metadatas

                st.success("Content indexed and ready!")

        if st.button("Clear Indexed Data"):
            if os.path.exists("chroma_db"):
                shutil.rmtree("chroma_db")
                st.success("Indexed data cleared!")
            else:
                st.info("No indexed data to clear.")

if __name__ == "__main__":
    main()

