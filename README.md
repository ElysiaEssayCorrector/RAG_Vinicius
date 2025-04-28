# Corretor de Redações com RAG

Este sistema utiliza Inteligência Artificial com RAG (Retrieval-Augmented Generation) para analisar, corrigir e atribuir notas a redações no estilo ENEM. O sistema avalia as cinco competências do ENEM, fornece feedback detalhado e sugere melhorias específicas.

## Características

- **Análise de aderência ao tema**: Verifica se a redação aborda corretamente o tema proposto
- **Avaliação das cinco competências do ENEM**:
  - Competência 1: Domínio da norma padrão (200 pontos)
  - Competência 2: Compreensão do tema (200 pontos)
  - Competência 3: Argumentação (200 pontos)
  - Competência 4: Coesão textual (200 pontos)
  - Competência 5: Proposta de intervenção (200 pontos)
- **Feedback detalhado**: Análise, pontos fortes, pontos fracos e sugestões de melhoria
- **Ferramentas adicionais**: Sugestão de estrutura, análise de repertório, correção gramatical
- **Interface amigável**: Interface web em Streamlit para fácil utilização

## Tecnologias utilizadas

- **RAG (Retrieval-Augmented Generation)**: Para recuperar conhecimento específico sobre redações ENEM
- **ChromaDB**: Como banco de dados vetorial para armazenar e recuperar informações
- **Anthropic Claude 3.5 Haiku**: Para análise e geração de feedback
- **OpenAI GPT-4o-mini**: Alternativa para análise e geração de feedback
- **Streamlit**: Para a interface web
- **Docling**: Para processar documentos PDF e convertê-los para markdown

## Estrutura do projeto

```
corretor-redacao-enem/
│
├── data/
│   ├── raw/                # PDFs originais
│   └── processed/          # Documentos processados em markdown
│
├── models/
│   └── vectordb/           # Índices FAISS e metadados
│
├── src/
│   ├── preprocessing.py    # Processamento de documentos e redações
│   ├── evaluation.py       # Lógica de avaliação das competências
│   ├── rag.py              # Sistema de recuperação e geração
│   ├── utils.py            # Funções auxiliares
│   └── config.py           # Configurações do sistema
│
├── app.py                  # Aplicação Streamlit
│
└── requirements.txt        # Dependências do projeto
```

## Instalação

1. Clone o repositório:
   ```
   git clone https://github.com/seu-usuario/corretor-redacao-enem.git
   cd corretor-redacao-enem
   ```

2. Crie e ative um ambiente virtual:
   ```
   python -m venv venv
   # No Windows
   venv\Scripts\activate
   # No Linux/Mac
   source venv/bin/activate
   ```

3. Instale as dependências:
   ```
   pip install -r requirements.txt
   ```

4. Configure as credenciais de API:
   Crie um arquivo `.env` na raiz do projeto com suas chaves de API:
   ```
   OPENAI_API_KEY=sua_chave_openai
   ANTHROPIC_API_KEY=sua_chave_anthropic
   ```

5. Coloque os documentos PDF na pasta `data/raw/`

6. Execute o pré-processamento para criar o banco de dados vetorial:
   ```
   python -m src.preprocessing
   ```

## Uso

1. Inicie a aplicação Streamlit:
   ```
   streamlit run app.py
   ```

2. Acesse a interface web em `http://localhost:8501`

3. Digite o tema da redação e cole o texto da redação a ser analisada

4. Clique em "Analisar Redação" e aguarde o processamento

5. Explore o feedback detalhado e a pontuação atribuída

## Créditos

Este projeto foi desenvolvido como uma ferramenta educacional para auxiliar estudantes na preparação para o ENEM.

## Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo LICENSE para detalhes.
