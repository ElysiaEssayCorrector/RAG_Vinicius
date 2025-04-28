# Modifique o import do FAISS para Chroma
from langchain_community.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from anthropic import Anthropic
from openai import OpenAI

class RedacaoEvaluator:
    def __init__(self, vector_db_path, use_anthropic=True):
        """
        Inicializa o avaliador de redações
        
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
        
        # Inicializa o cliente da API escolhida
        if use_anthropic:
            self.client = Anthropic()
            self.model = "claude-3-5-haiku-20240307"
        else:
            self.client = OpenAI()
            self.model = "gpt-4o-mini"
            
        self.use_anthropic = use_anthropic
    
    # O método recuperar_documentos precisa ser atualizado
    def recuperar_documentos(self, query, categorias=None, k=5):
        """
        Recupera documentos relevantes do banco vetorial
        
        Args:
            query: Consulta para busca
            categorias: Lista de categorias para filtrar (opcional)
            k: Número de documentos a recuperar
            
        Returns:
            Lista de documentos recuperados
        """
        if categorias:
            # Busca com filtro de categoria
            # No ChromaDB, precisamos adaptar a forma de filtragem
            filter_dict = {"category": {"$in": categorias}} if isinstance(categorias, list) else {"category": categorias}
            docs = self.vectorstore.similarity_search(
                query,
                k=k,
                filter=filter_dict
            )
        else:
            # Busca sem filtro
            docs = self.vectorstore.similarity_search(query, k=k)
        
        return docs
        
    # Resto da classe permanece igual