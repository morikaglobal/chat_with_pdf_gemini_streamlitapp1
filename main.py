import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# Read PDFs loaded and get all text from all pages in one text (context)
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


# FAISS
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details and you answer the question based on the context provided and not hallucinate.
    If the answer is not found in the provided context or if the context does not have enough information for you to answer the question, inform that the relevant context was not available in the given context and add 'it seems the relevant information is not available in uploaded PDFs to answer your question, please ask a question related to the contents of the PDFs uploaded.'
    
    \n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """


    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)

    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
   
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Answer: ", response["output_text"])


def main():
    st.set_page_config("Chat with PDF")
    st.header("Chat with PDFs using Gemini")

    user_question = st.text_input("Ask a question to answer from the PDF files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Upload your PDFs")
        pdf_docs = st.file_uploader("Upload the PDF files you want answers to your questions from and click on the Submit & Process Button \n If you do not have any PDFs in mind, try this:",
        "If you don't have any PDFs in mind, try this:",
        
        accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done, now it is ready to answer your question!")

print("ALL GOOD")


if __name__ == "__main__":
    main()