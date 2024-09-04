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
from collections import Counter

# Charger les variables d'environnement
load_dotenv()  # Assure que les variables d'environnement sont chargées depuis le fichier .env

# Accéder aux variables d'environnement
SUPABASE_HOST = os.getenv("SUPABASE_HOST")
SUPABASE_DB = os.getenv("SUPABASE_DB")
SUPABASE_PORT = os.getenv("SUPABASE_PORT")
SUPABASE_USER = os.getenv("SUPABASE_USER")
SUPABASE_PASSWORD = os.getenv("SUPABASE_PASSWORD")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Sentiment analysis (simple placeholder function)
def analyze_sentiment(text):
    # Placeholder function for sentiment analysis.
    return "positive" if "good" in text.lower() else "negative"

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
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)  # Pass the API key to OpenAIEmbeddings
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(api_key=OPENAI_API_KEY)  # Pass the API key to ChatOpenAI
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def analyze_conversation(text):
    lines = text.split('\n')
    participants = Counter()
    topics = Counter()
    sentiments = Counter()

    for line in lines:
        if ':' in line:
            person, message = line.split(':', 1)
            person = person.strip()
            message = message.strip()
            participants[person] += 1
            if "topic" in message.lower():
                topics[message] += 1
            sentiment = analyze_sentiment(message)
            sentiments[(person, sentiment)] += 1

    most_active = participants.most_common(1)[0] if participants else None
    most_positive = [p for p, s in sentiments if s == 'positive']
    most_negative = [p for p, s in sentiments if s == 'negative']

    return participants, topics, most_active, most_positive, most_negative

def handle_userinput(user_question):
    response = st.session_state.conversation.invoke({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            save_message_to_db(user_message=message.content, bot_response="")
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
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
    pass  # Implement this if needed

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
    st.set_page_config(page_title="Chat avec notre conversation WhatsApp", page_icon=":speech_balloon:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat avec notre conversation WhatsApp :speech_balloon:")

    with st.sidebar:
        st.subheader("Vos documents")
        pdf_docs = st.file_uploader("Téléchargez vos fichiers PDF ici et cliquez sur 'Process'", accept_multiple_files=True)
        if pdf_docs:
            st.write("Fichiers téléchargés :")
            for pdf in pdf_docs:
                st.write(pdf.name)

        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Traitement en cours"):
                    try:
                        raw_text = get_pdf_text(pdf_docs)
                        st.write("Texte extrait des PDFs :")
                        st.write(raw_text[:1000])  # Affichez un extrait du texte pour vérification

                        text_chunks = get_text_chunks(raw_text)
                        vectorstore = get_vectorstore(text_chunks)
                        st.session_state.conversation = get_conversation_chain(vectorstore)
                        st.write("Vectorstore créé et conversation chain initialisée.")

                        # Analyze conversation
                        participants, topics, most_active, most_positive, most_negative = analyze_conversation(raw_text)
                        
                        # Display summary
                        st.subheader("Résumé de la Conversation")
                        st.write(f"**Contexte de la conversation :** Le texte extrait des fichiers PDF.")
                        st.write(f"**Participants :** {', '.join(participants.keys())}")
                        st.write(f"**Sujets abordés :** {', '.join(topics.keys())}")
                        st.write(f"**Personne la plus active :** {most_active[0] if most_active else 'Aucun participant'}")
                        st.write(f"**Personnes les plus positives :** {', '.join(most_positive) if most_positive else 'Aucune'}")
                        st.write(f"**Personnes les plus négatives :** {', '.join(most_negative) if most_negative else 'Aucune'}")

                    except Exception as e:
                        st.error(f"Une erreur est survenue : {e}")
            else:
                st.warning("Veuillez télécharger des fichiers PDF avant de cliquer sur 'Process'")

    user_question = st.text_input("Posez une question sur votre conversation WhatsApp:")
    if user_question:
        handle_userinput(user_question)

if __name__ == '__main__':
    main()
