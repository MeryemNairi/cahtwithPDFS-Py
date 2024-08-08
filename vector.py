import os
import json
from supabase import create_client, Client
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st


SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

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
    vectors = [embeddings.embed_text(chunk) for chunk in text_chunks]
    
    # Save vectors to Supabase
    data = [{"document_id": str(i), "vector": json.dumps(vectors[i].tolist())} for i in range(len(vectors))]
    supabase.table("vector_store").insert(data).execute()
    
    # Load vectors from Supabase
    response = supabase.table("vector_store").select("vector").execute()
    stored_vectors = [json.loads(record['vector']) for record in response.data]
    
    vectorstore = FAISS.from_vectors(vectors=stored_vectors)
    return vectorstore

def display_vectorstore(vectorstore):
    st.write("Vector Store Details:")
    st.write(f"Number of documents: {len(vectorstore.docstore._dict)}")
    st.write("Sample Vectors:")
    for idx, doc in enumerate(vectorstore.docstore._dict.items()):
        if idx >= 5:  # Display only the first 5 vectors for brevity
            break
        st.write(f"Document ID: {doc[0]}")
        st.write(f"Vector: {vectorstore.vectors[idx][:10]}...")  # Display only the first 10 dimensions
