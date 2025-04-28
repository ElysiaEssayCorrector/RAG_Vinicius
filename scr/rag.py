import os
from langchain_community.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from anthropic import Anthropic
from openai import OpenAI

class RAGSystem:
    def __init__(self, vector_db_path, use_anthropic=True):
        """
        Inicializa o sistema RAG (Retrieval-Augmented Generation)
        
        Args:
            vector_db_path: Caminho para o banco de dados vetorial (Chroma)
            use_anthropic: Se True, usa Claude da Anthropic; se False, usa GPT da OpenAI
        """
        # Carrega o banco de dados Chroma
        embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma(
            persist_directory=vector_db_path,
            embedding_function=embeddings
        )
        
        # Resto do código permanece igual
        
    def retrieve(self, query, categoria=None, k=5):
        """
        Recupera documentos relevantes do banco de dados vetorial
        
        Args:
            query: Consulta para busca
            categoria: Categoria para filtrar (opcional)
            k: Número de documentos a recuperar
            
        Returns:
            Lista de documentos recuperados
        """
        if categoria:
            # Busca com filtro de categoria
            docs = self.vectorstore.similarity_search(
                query,
                k=k,
                filter={"category": categoria}
            )
        else:
            # Busca sem filtro
            docs = self.vectorstore.similarity_search(query, k=k)
        
        return docs
        
    # Resto da classe permanece igual