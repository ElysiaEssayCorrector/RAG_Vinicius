import os
import re
import docling
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import pickle
import numpy as np


def process_documents(raw_dir, processed_dir):
    """
    Processa documentos PDF para markdown
    
    Args:
        raw_dir: Diretório com PDFs brutos
        processed_dir: Diretório para salvar documentos processados
    """
    os.makedirs(processed_dir, exist_ok=True)
    
    # Listar arquivos PDF no diretório
    pdf_files = [f for f in os.listdir(raw_dir) if f.endswith('.pdf')]
    
    for pdf_file in pdf_files:
        input_path = os.path.join(raw_dir, pdf_file)
        output_path = os.path.join(processed_dir, f"{os.path.splitext(pdf_file)[0]}.md")
        
        # Verificar se o arquivo já foi processado
        if os.path.exists(output_path):
            print(f"Arquivo {output_path} já existe. Pulando...")
            continue
        
        print(f"Processando {pdf_file}...")
        
        try:
            # Extrair texto do PDF usando docling
            doc = docling.Document.from_pdf(input_path)
            markdown_text = doc.to_markdown()
            
            # Salvar como markdown
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_text)
                
            print(f"Processado com sucesso: {output_path}")
            
        except Exception as e:
            print(f"Erro ao processar {pdf_file}: {str(e)}")

def get_document_category(filename):
    """
    Determina a categoria do documento com base no nome do arquivo
    
    Args:
        filename: Nome do arquivo
        
    Returns:
        Categoria do documento
    """
    filename_lower = filename.lower()
    
    if "norma" in filename_lower or "gramatic" in filename_lower:
        return "norma_culta"
    elif "tema" in filename_lower or "compreens" in filename_lower:
        return "tema"
    elif "argument" in filename_lower:
        return "argumentacao"
    elif "coes" in filename_lower:
        return "coesao"
    elif "interven" in filename_lower or "proposta" in filename_lower:
        return "intervencao"
    elif "estrutura" in filename_lower:
        return "estrutura"
    elif "exemplo" in filename_lower or "introduc" in filename_lower or "desenvolv" in filename_lower or "conclus" in filename_lower:
        return "exemplos"
    else:
        return "geral"

def chunk_documents(processed_dir):
    """
    Divide documentos em chunks para vetorização
    
    Args:
        processed_dir: Diretório com documentos processados
        
    Returns:
        Tuple com lista de chunks e metadados
    """
    # Listar arquivos processados
    md_files = [f for f in os.listdir(processed_dir) if f.endswith('.md')]
    
    all_chunks = []
    all_metadatas = []
    
    # Inicializar o divisor de texto
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    for md_file in md_files:
        file_path = os.path.join(processed_dir, md_file)
        
        # Ler o conteúdo do arquivo
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Dividir em chunks
        chunks = text_splitter.split_text(content)
        
        # Determinar categoria com base no nome do arquivo
        category = get_document_category(md_file)
        
        # Adicionar metadados para cada chunk
        metadatas = [{"source": md_file, "category": category} for _ in chunks]
        
        all_chunks.extend(chunks)
        all_metadatas.extend(metadatas)
    
    print(f"Total de chunks gerados: {len(all_chunks)}")
    return all_chunks, all_metadatas

def create_chroma_db(chunks, metadatas, save_dir):
    """
    Cria um banco de dados Chroma a partir dos chunks e salva no diretório especificado
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Criar embeddings usando OpenAI - VERSÃO CORRIGIDA
    try:
        # Tente com a nova API
        embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    except TypeError:
        # Fallback para versão mais antiga se necessário
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