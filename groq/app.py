
import  streamlit as st
import os 
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv

load_dotenv()

os.environ['OPENAI_API_KEY']=os.getenv('OPENAI_API_KEY')
groq_api_key=os.getenv('GROQ_API_KEY')

st.title("ChatGroq with Llama3 demo")

llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="Llama3-8b-8192")

prompt=ChatPromptTemplate.from_template(
    """Answer the questions based on the provided context only.
    Please provide the most accurate response based on the questions
    <context>
    {context}
    <context>
    Questions:{input}
    """
)


def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=OpenAIEmbeddings()
        ## Data Ingestion
        st.session_state.loader=PyPDFDirectoryLoader('../huggingface/datamining')
        ## Documents Loading 
        st.session_state.docs=st.session_state.loader.load()
        ## Chunk Creation
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        # Doument split
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs)
        #Vector Store Creation
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)


prompt1=st.text_input("Enter Your Questions")

if st.button("Document Embedding"):
    vector_embedding()
    st.write('Vector Store is ready')


if prompt1:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)

    response=retrieval_chain.invoke({'input':prompt1})
    st.write(response['answer'])

    with st.expander('Documents Similarity Search'):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("_____________________________")


