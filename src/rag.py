import os
import json
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
        
        # Inicializa o cliente da API escolhida
        if use_anthropic:
            self.client = Anthropic()
            self.model = "claude-3-5-haiku-20240307"
        else:
            self.client = OpenAI()
            self.model = "gpt-4o-mini"
            
        self.use_anthropic = use_anthropic
    
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
            if isinstance(categoria, list):
                filter_dict = {"category": {"$in": categoria}}
            else:
                filter_dict = {"category": categoria}
                
            docs = self.vectorstore.similarity_search(
                query,
                k=k,
                filter=filter_dict
            )
        else:
            # Busca sem filtro
            docs = self.vectorstore.similarity_search(query, k=k)
        
        return docs
    
    def safe_api_call(self, prompt, attempt=0, max_attempts=3):
        """
        Executa chamada de API com tratamento de erros e tentativas
        
        Args:
            prompt: Texto do prompt
            attempt: Tentativa atual (para retry)
            max_attempts: Número máximo de tentativas
            
        Returns:
            Texto da resposta
        """
        try:
            if self.use_anthropic:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4000,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=4000
                )
                return response.choices[0].message.content
        except Exception as e:
            if attempt < max_attempts:
                # Backoff exponencial
                import time
                wait_time = 2 ** attempt
                print(f"Erro na API, tentando novamente em {wait_time}s: {str(e)}")
                time.sleep(wait_time)
                return self.safe_api_call(prompt, attempt + 1, max_attempts)
            else:
                raise Exception(f"Erro na API após {max_attempts} tentativas: {str(e)}")
    
    def sugerir_estrutura_redacao(self, tema):
        """
        Sugere uma estrutura para redação com base no tema
        
        Args:
            tema: Tema da redação
            
        Returns:
            Texto com sugestão de estrutura
        """
        # Recuperar documentos sobre estrutura de redação
        docs_estrutura = self.retrieve("estrutura redação enem introdução desenvolvimento conclusão", "estrutura", k=2)
        docs_exemplos = self.retrieve("exemplos redação enem", "exemplos", k=1)
        
        contexto = "\n\n".join([doc.page_content for doc in docs_estrutura + docs_exemplos])
        
        prompt = f"""
        Você é um especialista em redações do ENEM. Com base no tema fornecido, sugira uma estrutura detalhada para uma redação nota 1000.

        # Tema da redação:
        {tema}

        # Informações sobre estrutura de redação no ENEM:
        {contexto}

        Forneça:
        1. Uma sugestão de abordagem para o tema
        2. Uma estrutura para introdução (com repertório possível)
        3. Uma estrutura para cada parágrafo de desenvolvimento (com exemplos de argumentos)
        4. Uma estrutura para conclusão (com modelo de proposta de intervenção)

        Seja específico e considere o tema proposto.
        """
        
        # Chama a API
        resultado = self.safe_api_call(prompt)
        return resultado
    
    def analisar_repertorio(self, redacao_text):
        """
        Analisa o repertório sociocultural utilizado na redação
        
        Args:
            redacao_text: Texto da redação
            
        Returns:
            Texto com análise do repertório
        """
        # Recuperar documentos sobre repertório
        docs = self.retrieve("repertório sociocultural redação enem argumentação", ["argumentacao", "exemplos"], k=3)
        contexto = "\n\n".join([doc.page_content for doc in docs])
        
        prompt = f"""
        Você é um especialista em redações do ENEM. Analise o repertório sociocultural utilizado na redação abaixo.

        # Texto da redação:
        {redacao_text}

        # Informações sobre repertório sociocultural no ENEM:
        {contexto}

        Forneça:
        1. Identificação de todos os repertórios socioculturais utilizados (citações, referências, dados, exemplos históricos, etc.)
        2. Avaliação da qualidade e relevância de cada repertório
        3. Sugestões de repertórios adicionais que poderiam enriquecer a argumentação

        Seja específico e considere a pertinência dos repertórios ao tema da redação.
        """
        
        # Chama a API
        resultado = self.safe_api_call(prompt)
        return resultado