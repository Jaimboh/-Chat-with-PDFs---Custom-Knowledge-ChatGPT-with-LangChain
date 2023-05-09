import os
import pandas as pd
from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
import textract

# Retrieve the OpenAI API key from the Streamlit Secrets Manager
api_key = st.secrets["openai"]["api_key"]
# Set the OpenAI API key as an environment variable
os.environ["OPENAI_API_KEY"] = api_key

st.title("Document Query")

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    doc = textract.process(uploaded_file)
    with open('uploaded_file.txt', 'w') as f:
        f.write(doc.decode('utf-8'))
    with open('uploaded_file.txt', 'r') as f:
        text = f.read()

    query = st.text_input("Enter your query")

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text))

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    text_splitter = RecursiveCharacterTextSplitter(
        
        chunk_size = 512,
        chunk_overlap  = 24,
        length_function = count_tokens,
    )
    chunks = text_splitter.create_documents([text])
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(chunks, embeddings)
    docs = db.similarity_search(query)
    chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
    chain.run(input_documents=docs, question=query)

    if st.button("Run Query"):
        st.write(chain.get_output())
