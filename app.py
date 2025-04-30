import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

def get_vector_store(text_chunks):
    # 1) create embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # 2) build Chroma store (will create ./chroma_db/)
    vector_store = Chroma.from_texts(
        texts=text_chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    # 3) save to disk
    vector_store.persist()

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
    model = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro-latest", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context","question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

import time, random
from google.api_core.exceptions import ResourceExhausted

def run_with_retries(chain, inputs, max_retries=5):
    """
    Try `chain(...)` up to max_retries times, backing off on ResourceExhausted.
    Returns the chain output dict, or re-raises after max_retries.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return chain(inputs, return_only_outputs=True)
        except ResourceExhausted as err:
            # If the error includes a retry_delay, use it; otherwise exponential/backoff
            delay = getattr(err, 'retry_delay', None)
            wait = delay.seconds if delay and hasattr(delay, 'seconds') else (2 ** attempt)
            jitter = random.uniform(0, 1)
            total_wait = wait + jitter
            st.warning(
                f"Hit rate limit (attempt {attempt}/{max_retries}). "
                f"Retrying in {total_wait:.1f}s…"
            )
            time.sleep(total_wait)
    # all retries failed
    raise

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # load the persisted Chroma store
    db = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )
    docs = db.similarity_search(user_question)
    chain = get_conversational_chain()
    output = run_with_retries(chain, {"input_documents": docs, "question": user_question})
    st.write("Reply:", output["output_text"])

def main():
    st.set_page_config(page_title="Chat with Multiple PDF")
    st.header("Chat with multiple PDF using Gemini 💁")

    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click Submit & Process",
            accept_multiple_files=True
        )
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                chunks = get_text_chunks(raw_text)
                get_vector_store(chunks)
                st.success("Indexing complete!")

if __name__ == "__main__":
    main()

