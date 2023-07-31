import streamlit as st
import qdrant_client
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
#from langchain.vectorstores import FAISS
from langchain.vectorstores import Qdrant
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from qdrant_client.http import models

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

#nb changed below code to not take input text chunks as initialising from a pre-existing collection in Qdrant Cloud
def get_vector_store():
    #embeddings = OpenAIEmbeddings()
    
    #vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    client = qdrant_client.QdrantClient(
        os.getenv("QDRANT_HOST"),
        api_key=os.getenv("QDRANT_API_KEY")
    )

    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")

    vector_store = Qdrant(
    client=client, collection_name="instructor_collection", 
    embeddings=embeddings,
    )
    return vector_store


def get_conversation_chain(vector_store):
    #llm = ChatOpenAI()
    #llm = HuggingFaceHub(repo_id="meta-llama/Llama-2-7b-chat-hf", model_kwargs={"temperature":0.5, "max_length":512})
    llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature":0.1, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                #vectorstore = get_vectorstore(text_chunks)
                vector_store = get_vector_store()

                # create conversation chain
                #st.session_state.conversation = get_conversation_chain(
                 #   vectorstore)
                st.session_state.conversation = get_conversation_chain(
                    vector_store)


if __name__ == '__main__':
    main()
