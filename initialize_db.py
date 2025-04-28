"""
Script para inicializar o banco de dados vetorial.
Execute este script antes de iniciar a aplicação pela primeira vez.
"""
import os
import sys

# Adiciona o diretório atual ao caminho de busca do Python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, VECTOR_DB_PATH
from src.preprocessing import process_documents, chunk_documents, create_chroma_db

def main():
    # Verifica se os diretórios existem
    for path in [RAW_DATA_PATH, PROCESSED_DATA_PATH, VECTOR_DB_PATH]:
        os.makedirs(path, exist_ok=True)
    
    # Verifica se há documentos no diretório raw
    if not os.listdir(RAW_DATA_PATH):
        print(f"Nenhum documento encontrado em {RAW_DATA_PATH}. "
              f"Por favor, adicione os arquivos PDF antes de continuar.")
        return
    
    # Processa os documentos
    print("Processando documentos PDF...")
    process_documents(RAW_DATA_PATH, PROCESSED_DATA_PATH)
    
    # Divide em chunks e obtém metadados
    print("Dividindo documentos em chunks...")
    chunks, metadatas = chunk_documents(PROCESSED_DATA_PATH)
    
    # Cria banco de dados Chroma
    print("Criando banco de dados Chroma...")
    create_chroma_db(chunks, metadatas, VECTOR_DB_PATH)
    
    print("\nInicialização concluída com sucesso!")
    print(f"Banco de dados vetorial criado em: {VECTOR_DB_PATH}")
    print("Agora você pode iniciar a aplicação com: streamlit run app.py")

if __name__ == "__main__":
    main()