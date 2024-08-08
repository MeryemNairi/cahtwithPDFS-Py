import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import os
import psycopg2
import traceback

# Charger les variables d'environnement
load_dotenv()
SUPABASE_HOST = os.getenv("SUPABASE_HOST")
SUPABASE_DB = os.getenv("SUPABASE_DB")
SUPABASE_PORT = os.getenv("SUPABASE_PORT")
SUPABASE_USER = os.getenv("SUPABASE_USER")
SUPABASE_PASSWORD = os.getenv("SUPABASE_PASSWORD")

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

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation.invoke({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            # Save user message to the database
            save_message_to_db(user_message=message.content, bot_response="")
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            # Update database with bot response
            update_message_in_db(bot_response=message.content)

def get_db_connection():
    conn = psycopg2.connect(
        host=SUPABASE_HOST,
        dbname=SUPABASE_DB,
        user=SUPABASE_USER,
        password=SUPABASE_PASSWORD,
        port=SUPABASE_PORT
    )
    return conn

def save_message_to_db(user_message, bot_response):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO chat_history_Files (user_message, bot_response) VALUES (%s, %s)",
            (user_message, bot_response)
        )
        conn.commit()
        cursor.close()
        conn.close()
        print("Message saved to database.")
    except Exception as e:
        print(f"Error saving message to database: {e}")
        traceback.print_exc()

def update_message_in_db(bot_response):
    # Implement this if needed
    pass

def test_db_connection():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        test_data = {
            "user_message": "Test message",
            "bot_response": "Test response"
        }
        cursor.execute(
            "INSERT INTO chat_history_Files (user_message, bot_response) VALUES (%s, %s)",
            (test_data["user_message"], test_data["bot_response"])
        )
        conn.commit()
        cursor.close()
        conn.close()
        print("Test message saved to database.")
    except Exception as e:
        print(f"Test Error: {e}")
        traceback.print_exc()

def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
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
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if pdf_docs:
            st.write("Fichiers téléchargés :")
            for pdf in pdf_docs:
                st.write(pdf.name)

        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing"):
                    try:
                        raw_text = get_pdf_text(pdf_docs)
                        st.write("Texte extrait des PDFs :")
                        st.write(raw_text[:1000])  # Affichez un extrait du texte pour vérification

                        text_chunks = get_text_chunks(raw_text)
                        vectorstore = get_vectorstore(text_chunks)
                        st.session_state.conversation = get_conversation_chain(vectorstore)
                        st.write("Vectorstore créé et conversation chain initialisée.")
                    except Exception as e:
                        st.error(f"Une erreur est survenue : {e}")
            else:
                st.warning("Veuillez télécharger des fichiers PDF avant de cliquer sur 'Process'")

if __name__ == '__main__':
    main()
