from pinecone import Pinecone
from langchain_community.vectorstores import Pinecone as PineconeStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from pypdf import PdfReader
from langchain.schema import Document



import streamlit as st
from langchain_community.vectorstores import FAISS


# Assuming this function encodes the question into a vector representation
def encode_question(question,embeddings):
    question_vector = embeddings.embed_query(question)  # Encode the question into a vector
    return question_vector

def save_vector_store(text_chunks,embeddings):

    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    new_db = FAISS.load_local("faiss_index_V2", embeddings, allow_dangerous_deserialization=True)
    new_db.merge_from(vectorstore)
    new_db.save_local('faiss_index_V2')

    return st.write("vector Store is Saved")


def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def create_docs(user_pdf_list, unique_id):
    docs=[]
    for filename in user_pdf_list:
        
        chunks=get_pdf_text(filename)

        #Adding items to our list - Adding data & its metadata
        docs.append(Document(
            page_content=chunks,
            metadata={"name": filename.name,"id":filename.file_id,"type=":filename.type,"size":filename.size,"unique_id":unique_id},
        ))

    return docs





def push_to_pinecone(pinecone_apikey,pinecone_environment,pinecone_index_name,embeddings,docs):
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(docs)
    pinecone = Pinecone(
        api_key=pinecone_apikey,environment=pinecone_environment
        )
    # create a vectorstore from the chunks
    vector_store=PineconeStore.from_documents(document_chunks,embeddings,index_name=pinecone_index_name)


def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()
    retriever = vector_store.as_retriever()  
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    # If there is no chat_history, then the input is just passed directly to the retriever. 
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

    
def get_conversational_rag_chain(retriever_chain): 
    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages([
      ("system", "you are an AI Chatbot who first ask user that which language he will prefer for example HINDI, ENGLISH, or HINGLISH(if hinglish-for exmaple, namaste me aapke liye kya kr skta hu  ) for communication. than you will ask him options like who he is JOB SEEKER, RECRUTERM or INSTITUTE, when they enter their identity , if the user is 'Job Seaker' than you will ask him in which sector he want to do job, if the person is 'recrutor' than you will ask in how many person do you want and for which role, if ' institude' you will ask them to how many students do you have and what are their major skills and ask them to upload their students resume:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    # for passing a list of Documents to a model.
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

