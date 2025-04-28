import os
from dotenv import load_dotenv

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# Configurações da API OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o-mini"  # Modelo padrão

# Configurações da API Anthropic
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = "claude-3-5-haiku-20240307"  # Modelo padrão

# Configurações do sistema
VECTOR_DB_PATH = "models/vectordb"
RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"

# Configurações de divisão de texto
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Configurações de avaliação
COMPETENCIAS = {
    "competencia_1": {
        "nome": "Domínio da norma padrão",
        "descricao": "Demonstrar domínio da norma padrão da língua escrita.",
        "peso": 200
    },
    "competencia_2": {
        "nome": "Compreensão do tema",
        "descricao": "Compreender a proposta de redação e aplicar conceitos das várias áreas de conhecimento para desenvolver o tema.",
        "peso": 200
    },
    "competencia_3": {
        "nome": "Argumentação",
        "descricao": "Selecionar, relacionar, organizar e interpretar informações, fatos, opiniões e argumentos em defesa de um ponto de vista.",
        "peso": 200
    },
    "competencia_4": {
        "nome": "Coesão textual",
        "descricao": "Demonstrar conhecimento dos mecanismos linguísticos necessários para a construção da argumentação.",
        "peso": 200
    },
    "competencia_5": {
        "nome": "Proposta de intervenção",
        "descricao": "Elaborar proposta de intervenção para o problema abordado, respeitando os direitos humanos.",
        "peso": 200
    }
}