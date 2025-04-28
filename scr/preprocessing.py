import os
import re
import docling
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
import chromadb
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings
import pickle
import numpy as np

# Resto das funções permanece igual até create_chroma_db

def create_chroma_db(chunks, metadatas, save_dir):
    """
    Cria um banco de dados Chroma a partir dos chunks e salva no diretório especificado
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Criar embeddings usando OpenAI
    embeddings = OpenAIEmbeddings()
    
    # Criar e salvar banco de dados Chroma
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=save_dir
    )
    
    # Persistir o banco de dados
    vectorstore.persist()
    
    print(f"Banco de dados Chroma criado e salvo em {save_dir}")

def main():
    # Diretórios de trabalho
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    vector_dir = "models/vectordb"
    
    # Processar documentos PDF para markdown
    process_documents(raw_dir, processed_dir)
    
    # Dividir em chunks e obter metadados
    chunks, metadatas = chunk_documents(processed_dir)
    
    # Criar banco de dados Chroma
    create_chroma_db(chunks, metadatas, vector_dir)

if __name__ == "__main__":
    main()