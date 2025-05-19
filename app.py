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

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# HuggingFace embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Set up Groq model (Mixtral)
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

# Split text
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

# Create vector store
def get_vector_store(text_chunks):
    if os.path.exists("chroma_db"):
        shutil.rmtree("chroma_db")
    vector_store = Chroma.from_texts(
        texts=text_chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    vector_store.persist()

# Handle user query
def user_input(user_question):
    db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    docs = db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply:", response["output_text"])

# Streamlit UI
def main():
    st.set_page_config(page_title="Chat with Multiple PDFs (Groq)")
    st.header("Chat with multiple PDFs using Mixtral 💬")

    user_question = st.text_input("Ask a question from the uploaded PDFs:")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Upload & Process PDFs:")
        pdf_docs = st.file_uploader("Upload PDF files", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                chunks = get_text_chunks(raw_text)
                get_vector_store(chunks)
                st.success("PDFs indexed and ready!")

if __name__ == "__main__":
    main()
