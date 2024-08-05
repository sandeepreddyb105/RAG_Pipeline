import pandas as pd
import streamlit as st   
import warnings
warnings.filterwarnings('ignore')

import tiktoken
import os
import shutil
from langchain.chat_models import ChatOpenAI
import os
import chromadb
import openai

from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

st.set_page_config(
    page_title="LLM model",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items= {
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"})


#summaries data
vector_db_folder = "./vector_db"
vector_db_name = "concepts_of_biology"


def embedding_model():
    # Retrieve embedding function from code env resources
    emb_model = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=emb_model,
        cache_folder=os.getenv('SENTENCE_TRANSFORMERS_HOME')
    )
    
    return emb_model, embeddings
    
    
def load_vector_db(vector_db_name, vector_db_folder, embeddings):
    # Load vector database

    persist_dir = os.path.join(vector_db_folder, vector_db_name)
    vector_db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    
    return vector_db


os.environ["OPENAI_API_KEY"]= "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"
   

a,b = st.columns((4,1))

with a:
    
    a.markdown("<h3 style='text-align: center; color: black;'>Quantiphi Rag Implementation</h3>", unsafe_allow_html=True)
    query = st.text_input('Enter the question that you would like to ask LLM model')
    true =  st.button('Submit') 
            

    if true:
        emb_model, embeddings = embedding_model()
        vector_db = load_vector_db(cases_vector_db_name, cases_db_folder, embeddings)
        
        llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")
        q = query
        v = vector_db.similarity_search(q, k=4,include_metadata=True)
        # Run the chain by passing the output of the similarity search
        chain = load_qa_chain(llm, chain_type="stuff")
        res = chain({"input_documents": v, "question": q})
        output = res["output_text"]
        st.write(output)