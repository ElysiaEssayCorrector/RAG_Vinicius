import streamlit as st
import pandas as pd
import os
import sys
import re
import time

# Adiciona o diretório atual ao caminho de busca do Python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Agora importe dos módulos src
from src.evaluation import RedacaoEvaluator
from src.rag import RAGSystem

# Configuração da página
st.set_page_config(
    page_title="Corretor de Redações ENEM",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inicialização de sessão
if 'avaliacao' not in st.session_state:
    st.session_state.avaliacao = None
if 'redacao_submetida' not in st.session_state:
    st.session_state.redacao_submetida = False
if 'redacao_text' not in st.session_state:
    st.session_state.redacao_text = ""
if 'tema' not in st.session_state:
    st.session_state.tema = ""
if 'verificacao_tema' not in st.session_state:
    st.session_state.verificacao_tema = None

# Função para carregar os avaliadores (executada apenas uma vez)
@st.cache_resource
def carregar_avaliadores():
    # Paths para o banco de dados vetorial
    vector_db_path = "models/vectordb"
    
    # Verifica se o banco de dados existe
    if not os.path.exists(vector_db_path):
        st.error("Banco de dados vetorial não encontrado. Execute o script de pré-processamento primeiro.")
        return None, None
    
    # Carrega os avaliadores
    try:
        evaluator = RedacaoEvaluator(vector_db_path, use_anthropic=True)
        rag_system = RAGSystem(vector_db_path, use_anthropic=True)
        return evaluator, rag_system
    except Exception as e:
        st.error(f"Erro ao carregar avaliadores: {str(e)}")
        return None, None

# Função para processar a submissão
def processar_submissao():
    st.session_state.redacao_submetida = True
    st.session_state.redacao_text = redacao_text
    st.session_state.tema = tema
    
    # Verificação de aderência ao tema
    with st.spinner("Verificando aderência ao tema..."):
        verificacao = evaluator.verificar_aderencia_tema(redacao_text, tema)
        st.session_state.verificacao_tema = verificacao
    
    # Se a redação não fugir totalmente ao tema, continua a avaliação
    if verificacao["aderencia"] != "Fuga ao tema":
        with st.spinner("Analisando redação... (isso pode levar alguns minutos)"):
            resultado = evaluator.evaluate_redacao(redacao_text, tema)
            st.session_state.avaliacao = resultado

# Carrega os avaliadores
evaluator, rag_system = carregar_avaliadores()

# Título e descrição
st.title("📝 Corretor de Redações ENEM")
st.markdown("""
Este sistema utiliza inteligência artificial para analisar e corrigir redações no estilo ENEM, 
oferecendo feedback detalhado sobre cada uma das cinco competências avaliadas no exame.
""")

# Barra lateral com informações
with st.sidebar:
    st.header("Sobre o Corretor")
    st.markdown("""
    ### Como funciona?
    
    1. Digite o tema da redação
    2. Cole ou digite o texto da sua redação
    3. Clique em "Analisar Redação"
    4. Receba feedback detalhado e uma pontuação estimada
    
    ### Competências avaliadas
    
    1. **Domínio da norma padrão** (200 pontos)
    2. **Compreensão do tema** (200 pontos)
    3. **Argumentação** (200 pontos)
    4. **Coesão textual** (200 pontos)
    5. **Proposta de intervenção** (200 pontos)
    
    *Este sistema utiliza tecnologia RAG (Retrieval-Augmented Generation) e modelos de IA avançados para fornecer feedback baseado nas diretrizes oficiais do ENEM.*
    """)
    
    # Opções avançadas
    st.subheader("Opções Avançadas")
    modelo = st.radio(
        "Modelo de IA",
        options=["Claude (Anthropic)", "GPT-4o-mini (OpenAI)"],
        index=0
    )
    
    # Botões de ferramentas adicionais
    st.subheader("Ferramentas Adicionais")
    ferramentas_expander = st.expander("Expandir Ferramentas")
    with ferramentas_expander:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Sugerir Estrutura"):
                if 'tema' in st.session_state and st.session_state.tema:
                    with st.spinner("Gerando sugestão de estrutura..."):
                        sugestao = rag_system.sugerir_estrutura_redacao(st.session_state.tema)
                        st.session_state.sugestao_estrutura = sugestao
                else:
                    st.warning("Por favor, informe o tema da redação primeiro.")
        
        with col2:
            if st.button("Analisar Repertório"):
                if 'redacao_text' in st.session_state and st.session_state.redacao_text:
                    with st.spinner("Analisando repertório sociocultural..."):
                        analise = rag_system.identificar_repertorio_sociocultural(st.session_state.redacao_text)
                        st.session_state.analise_repertorio = analise
                else:
                    st.warning("Por favor, submeta uma redação primeiro.")

# Formulário principal
with st.form(key="redacao_form"):
    tema = st.text_input("Tema da Redação", 
                         help="Digite o tema oficial da redação, exatamente como foi proposto")
    
    redacao_text = st.text_area("Texto da Redação", 
                               height=300,
                               help="Cole ou digite o texto completo da redação a ser analisada")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        submit_button = st.form_submit_button(label="Analisar Redação")

# Processa a submissão quando o botão é pressionado
if submit_button:
    if len(redacao_text.strip()) < 100:
        st.error("O texto da redação é muito curto. Por favor, insira uma redação completa.")
    elif len(tema.strip()) < 10:
        st.error("Por favor, insira o tema completo da redação.")
    else:
        processar_submissao()

# Exibe as ferramentas adicionais se solicitadas
if 'sugestao_estrutura' in st.session_state:
    with st.expander("Sugestão de Estrutura para Redação", expanded=True):
        st.markdown(st.session_state.sugestao_estrutura)
        if st.button("Fechar Sugestão", key="fechar_sugestao"):
            del st.session_state.sugestao_estrutura

if 'analise_repertorio' in st.session_state:
    with st.expander("Análise de Repertório Sociocultural", expanded=True):
        st.markdown(st.session_state.analise_repertorio)
        if st.button("Fechar Análise", key="fechar_analise"):
            del st.session_state.analise_repertorio

# Exibe resultados de verificação de tema
if st.session_state.verificacao_tema:
    verificacao = st.session_state.verificacao_tema
    
    if verificacao["aderencia"] == "Adequada":
        st.success(f"✅ **Aderência ao tema**: {verificacao['aderencia']}")
    elif verificacao["aderencia"] == "Tangenciamento":
        st.warning(f"⚠️ **Aderência ao tema**: {verificacao['aderencia']}")
    else:
        st.error(f"❌ **Aderência ao tema**: {verificacao['aderencia']}")
    
    with st.expander("Ver detalhes da análise de tema"):
        st.markdown("### Justificativa")
        st.markdown(verificacao["justificativa"])
        
        if verificacao["recomendacoes"]:
            st.markdown("### Recomendações")
            st.markdown(verificacao["recomendacoes"])
    
    # Se for fuga ao tema, para a análise aqui
    if verificacao["aderencia"] == "Fuga ao tema":
        st.error("A redação apresenta fuga ao tema. Corrija o texto para obter uma análise completa.")
        if st.button("Tentar Novamente"):
            st.session_state.redacao_submetida = False
            st.session_state.verificacao_tema = None
            st.experimental_rerun()

# Exibe resultados da avaliação
if st.session_state.avaliacao:
    avaliacao = st.session_state.avaliacao
    
    # Resumo da pontuação
    st.header("Resultado da Avaliação")
    
    # Exibe a nota total com um visual adequado
    nota_final = avaliacao["nota_final"]
    if nota_final >= 800:
        st.success(f"# Nota Final: {nota_final}/1000")
    elif nota_final >= 600:
        st.info(f"# Nota Final: {nota_final}/1000")
    elif nota_final >= 400:
        st.warning(f"# Nota Final: {nota_final}/1000")
    else:
        st.error(f"# Nota Final: {nota_final}/1000")
    
    # Cria um dataframe para exibir a pontuação por competência
    competencias_data = []
    for comp_id, comp_info in avaliacao["competencias"].items():
        nome_comp = ""
        if comp_id == "competencia_1":
            nome_comp = "Domínio da norma padrão"
        elif comp_id == "competencia_2":
            nome_comp = "Compreensão do tema"
        elif comp_id == "competencia_3":
            nome_comp = "Argumentação"
        elif comp_id == "competencia_4":
            nome_comp = "Coesão textual"
        elif comp_id == "competencia_5":
            nome_comp = "Proposta de intervenção"
        
        competencias_data.append({
            "Competência": nome_comp,
            "Pontuação": comp_info["pontuacao"],
            "Máximo": 200
        })
    
    # Cria e exibe um dataframe com as pontuações
    df = pd.DataFrame(competencias_data)
    
    # Cria um gráfico com as pontuações
    st.bar_chart(df.set_index("Competência")["Pontuação"])
    
    # Exibe as avaliações por competência
    st.subheader("Avaliação por Competência")
    
    # Cria abas para cada competência
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Competência 1", 
        "Competência 2", 
        "Competência 3", 
        "Competência 4", 
        "Competência 5"
    ])
    
    # Competência 1
    with tab1:
        comp = avaliacao["competencias"]["competencia_1"]
        st.markdown(f"### Domínio da norma padrão - {comp['pontuacao']}/200")
        
        st.markdown("#### Análise")
        st.markdown(comp["analise"])
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Pontos Fortes")
            st.markdown(comp["pontos_fortes"])
        
        with col2:
            st.markdown("#### Pontos Fracos")
            st.markdown(comp["pontos_fracos"])
        
        st.markdown("#### Sugestões de Melhoria")
        st.markdown(comp["sugestoes"])
    
    # Competência 2
    with tab2:
        comp = avaliacao["competencias"]["competencia_2"]
        st.markdown(f"### Compreensão do tema - {comp['pontuacao']}/200")
        
        st.markdown("#### Análise")
        st.markdown(comp["analise"])
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Pontos Fortes")
            st.markdown(comp["pontos_fortes"])
        
        with col2:
            st.markdown("#### Pontos Fracos")
            st.markdown(comp["pontos_fracos"])
        
        st.markdown("#### Sugestões de Melhoria")
        st.markdown(comp["sugestoes"])
    
    # Competência 3
    with tab3:
        comp = avaliacao["competencias"]["competencia_3"]
        st.markdown(f"### Argumentação - {comp['pontuacao']}/200")
        
        st.markdown("#### Análise")
        st.markdown(comp["analise"])
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Pontos Fortes")
            st.markdown(comp["pontos_fortes"])
        
        with col2:
            st.markdown("#### Pontos Fracos")
            st.markdown(comp["pontos_fracos"])
        
        st.markdown("#### Sugestões de Melhoria")
        st.markdown(comp["sugestoes"])
    
    # Competência 4
    with tab4:
        comp = avaliacao["competencias"]["competencia_4"]
        st.markdown(f"### Coesão textual - {comp['pontuacao']}/200")
        
        st.markdown("#### Análise")
        st.markdown(comp["analise"])
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Pontos Fortes")
            st.markdown(comp["pontos_fortes"])
        
        with col2:
            st.markdown("#### Pontos Fracos")
            st.markdown(comp["pontos_fracos"])
        
        st.markdown("#### Sugestões de Melhoria")
        st.markdown(comp["sugestoes"])
    
    # Competência 5
    with tab5:
        comp = avaliacao["competencias"]["competencia_5"]
        st.markdown(f"### Proposta de intervenção - {comp['pontuacao']}/200")
        
        st.markdown("#### Análise")
        st.markdown(comp["analise"])
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Pontos Fortes")
            st.markdown(comp["pontos_fortes"])
        
        with col2:
            st.markdown("#### Pontos Fracos")
            st.markdown(comp["pontos_fracos"])
        
        st.markdown("#### Sugestões de Melhoria")
        st.markdown(comp["sugestoes"])
    
    # Avaliação geral
    st.subheader("Avaliação Geral")
    
    avaliacao_geral = avaliacao["avaliacao_geral"]
    
    st.markdown("#### Análise Geral")
    st.markdown(avaliacao_geral["avaliacao_geral"])
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Competência Mais Forte")
        st.markdown(avaliacao_geral["competencia_mais_forte"])
    
    with col2:
        st.markdown("#### Competência Mais Fraca")
        st.markdown(avaliacao_geral["competencia_mais_fraca"])
    
    st.markdown("#### Sugestões Prioritárias")
    st.markdown(avaliacao_geral["sugestoes_prioritarias"])
    
    st.markdown("#### Conclusão")
    st.markdown(avaliacao_geral["conclusao"])
    
    # Botão para redefinir e fazer nova análise
    if st.button("Analisar Nova Redação"):
        st.session_state.redacao_submetida = False
        st.session_state.avaliacao = None
        st.session_state.verificacao_tema = None
        st.experimental_rerun()

# Configurações de estilo CSS
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        background-color: #f0f2f6;
    }

    .stTabs [aria-selected="true"] {
        background-color: #e6f0ff;
        color: #0066cc;
    }
    
    h3 {
        padding-top: 16px;
        padding-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)