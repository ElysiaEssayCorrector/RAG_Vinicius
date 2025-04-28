import re
import os
import json
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Baixa recursos necessários do NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def contar_palavras(texto):
    """
    Conta o número de palavras em um texto
    
    Args:
        texto: Texto a ser analisado
        
    Returns:
        Número de palavras
    """
    palavras = word_tokenize(texto, language='portuguese')
    return len(palavras)

def contar_caracteres(texto):
    """
    Conta o número de caracteres em um texto
    
    Args:
        texto: Texto a ser analisado
        
    Returns:
        Número de caracteres
    """
    return len(texto)

def contar_frases(texto):
    """
    Conta o número de frases em um texto
    
    Args:
        texto: Texto a ser analisado
        
    Returns:
        Número de frases
    """
    frases = sent_tokenize(texto, language='portuguese')
    return len(frases)

def contar_paragrafos(texto):
    """
    Conta o número de parágrafos em um texto
    
    Args:
        texto: Texto a ser analisado
        
    Returns:
        Número de parágrafos
    """
    paragrafos = [p for p in texto.split('\n') if p.strip()]
    return len(paragrafos)

def extrair_introducao(texto):
    """
    Extrai o primeiro parágrafo (introdução) de um texto
    
    Args:
        texto: Texto a ser analisado
        
    Returns:
        Texto da introdução
    """
    paragrafos = [p for p in texto.split('\n') if p.strip()]
    if paragrafos:
        return paragrafos[0]
    return ""

def extrair_desenvolvimento(texto):
    """
    Extrai os parágrafos de desenvolvimento (entre introdução e conclusão)
    
    Args:
        texto: Texto a ser analisado
        
    Returns:
        Texto do desenvolvimento
    """
    paragrafos = [p for p in texto.split('\n') if p.strip()]
    if len(paragrafos) <= 2:  # Se não houver parágrafos suficientes
        return ""
    return "\n".join(paragrafos[1:-1])

def extrair_conclusao(texto):
    """
    Extrai o último parágrafo (conclusão) de um texto
    
    Args:
        texto: Texto a ser analisado
        
    Returns:
        Texto da conclusão
    """
    paragrafos = [p for p in texto.split('\n') if p.strip()]
    if len(paragrafos) >= 2:
        return paragrafos[-1]
    return ""

def extrair_proposta_intervencao(texto):
    """
    Tenta extrair a proposta de intervenção da conclusão
    
    Args:
        texto: Texto da conclusão
        
    Returns:
        Texto da proposta de intervenção (ou toda a conclusão)
    """
    # Palavras-chave que podem indicar uma proposta de intervenção
    keywords = [
        "portanto", "logo", "assim", "dessa forma", "diante disso", "nesse sentido",
        "por fim", "em síntese", "em suma", "concluindo", "enfim"
    ]
    
    # Procura por frases que começam com palavras-chave
    for keyword in keywords:
        match = re.search(f"({keyword}[^.!?]*[.!?])", texto, re.IGNORECASE)
        if match:
            # Retorna o trecho da frase em diante
            start_idx = match.start()
            return texto[start_idx:]
    
    # Se não encontrar, retorna a conclusão inteira
    return texto

def identificar_repertorio_sociocultural(texto):
    """
    Tenta identificar possíveis elementos de repertório sociocultural
    
    Args:
        texto: Texto a ser analisado
        
    Returns:
        Lista de possíveis elementos de repertório
    """
    # Padrões para identificar citações, menções a autores, dados estatísticos, etc.
    padroes = [
        r'segundo\s+([^,.]+)',  # "Segundo Fulano"
        r'de acordo com\s+([^,.]+)',  # "De acordo com Fulano"
        r'conforme\s+([^,.]+)',  # "Conforme Fulano"
        r'([0-9]+%)',  # Porcentagens
        r'([0-9]+\s*(?:de cada|em cada)\s*[0-9]+)',  # "X de cada Y"
        r'"([^"]+)"',  # Citações entre aspas
        r"'([^']+)'",  # Citações entre aspas simples (CORRIGIDO)
        r'lei(?:\s+n[°º]?\s*[0-9.]+)',  # Menções a leis
        r'(?:constituição|carta magna)',  # Menções à Constituição
        r'(?:onu|unesco|unicef|oms|ibge)',  # Menções a organizações
    ]
    
    repertorio = []
    for padrao in padroes:
        matches = re.finditer(padrao, texto, re.IGNORECASE)
        for match in matches:
            # Extrai a frase completa onde o elemento foi encontrado
            start = max(0, texto.rfind('.', 0, match.start()) + 1)
            end = texto.find('.', match.end())
            if end == -1:
                end = len(texto)
            
            frase = texto[start:end].strip()
            if frase and frase not in repertorio:
                repertorio.append(frase)
    
    return repertorio

def salvar_avaliacao(avaliacao, output_dir="avaliacoes"):
    """
    Salva uma avaliação em formato JSON
    
    Args:
        avaliacao: Dicionário com os resultados da avaliação
        output_dir: Diretório para salvar a avaliação
        
    Returns:
        Path do arquivo salvo
    """
    # Cria o diretório se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Gera um nome de arquivo único baseado na data/hora
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"avaliacao_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Salva a avaliação
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(avaliacao, f, ensure_ascii=False, indent=4)
    
    return filepath

def carregar_avaliacao(filepath):
    """
    Carrega uma avaliação salva em formato JSON
    
    Args:
        filepath: Path do arquivo
        
    Returns:
        Dicionário com a avaliação
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        avaliacao = json.load(f)
    
    return avaliacao

def otimizar_prompt(redacao_text, max_tokens=4000):
    """
    Otimiza o texto da redação para economizar tokens
    
    Args:
        redacao_text: Texto da redação
        max_tokens: Número máximo de tokens
        
    Returns:
        Texto otimizado
    """
    # Estimativa básica de tokens (aproximadamente 4 caracteres por token)
    tokens_estimados = len(redacao_text) / 4
    
    # Se estiver dentro do limite, retornar como está
    if tokens_estimados <= max_tokens:
        return redacao_text
    
    # Caso contrário, reduzir preservando introdução e conclusão
    introducao = extrair_introducao(redacao_text)
    conclusao = extrair_conclusao(redacao_text)
    
    # Extrair desenvolvimento e reduzir se necessário
    desenvolvimento = redacao_text[len(introducao):len(redacao_text)-len(conclusao)]
    
    # Reduzir desenvolvimento para caber nos tokens disponíveis
    tokens_disponiveis = max_tokens - (len(introducao) + len(conclusao)) / 4
    chars_desenvolvimento = int(tokens_disponiveis * 4)
    
    if len(desenvolvimento) > chars_desenvolvimento:
        # Preservar início e fim do desenvolvimento
        metade = chars_desenvolvimento // 2
        desenvolvimento_reduzido = desenvolvimento[:metade] + "..." + desenvolvimento[-metade:]
        return introducao + desenvolvimento_reduzido + conclusao
    
    return redacao_text