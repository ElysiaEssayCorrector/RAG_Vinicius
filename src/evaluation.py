import json
import os
from langchain_community.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from anthropic import Anthropic
from openai import OpenAI
from src.utils import extrair_introducao, extrair_desenvolvimento, extrair_conclusao, extrair_proposta_intervencao

class RedacaoEvaluator:
    def __init__(self, vector_db_path, use_anthropic=True):
        """
        Inicializa o avaliador de redações
        """
        # Carrega o banco de dados Chroma - VERSÃO CORRIGIDA
        try:
            embeddings = OpenAIEmbeddings(
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
        except TypeError:
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
            if isinstance(categorias, list):
                filter_dict = {"category": {"$in": categorias}}
            else:
                filter_dict = {"category": categorias}
                
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
    
    def verificar_aderencia_tema(self, redacao_text, tema):
        """
        Verifica se a redação adere ao tema proposto
        
        Args:
            redacao_text: Texto da redação
            tema: Tema proposto
            
        Returns:
            Dicionário com avaliação
        """
        # Recuperar documentos sobre compreensão de tema
        docs = self.recuperar_documentos("compreensão tema redação enem", "tema", k=3)
        contexto = "\n\n".join([doc.page_content for doc in docs])
        
        prompt = f"""
        Você é um especialista em correção de redações do ENEM. Avalie se a redação abaixo está de acordo com o tema proposto.

        # Tema da redação:
        {tema}

        # Texto da redação:
        {redacao_text}

        # Contexto sobre avaliação de tema no ENEM:
        {contexto}

        Classifique a aderência ao tema em uma das categorias:
        1. "Adequada" - quando aborda o tema corretamente
        2. "Tangenciamento" - quando aborda o tema parcialmente
        3. "Fuga ao tema" - quando não aborda o tema proposto

        Forneça sua análise em formato JSON:
        ```json
        {{
          "aderencia": "Adequada|Tangenciamento|Fuga ao tema",
          "justificativa": "Explicação detalhada sobre a classificação",
          "recomendacoes": "Sugestões para melhorar a aderência ao tema"
        }}
        ```
        
        Responda SOMENTE com o JSON solicitado.
        """
        
        # Chama a API
        resultado = self.safe_api_call(prompt)
        
        # Extrair o JSON da resposta
        try:
            # Procura pelo JSON entre ```json e ```
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', resultado, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = resultado
            
            # Carrega o JSON
            resultado = json.loads(json_str)
            return resultado
        except Exception as e:
            raise Exception(f"Erro ao processar resultado da verificação de tema: {str(e)}\nResposta: {resultado}")
    
    def evaluate_redacao(self, redacao_text, tema):
        """
        Avalia uma redação completa em todos os critérios
        
        Args:
            redacao_text: Texto da redação
            tema: Tema proposto
            
        Returns:
            Dicionário com avaliação completa
        """
        # Verificar aderência ao tema primeiro
        verificacao = self.verificar_aderencia_tema(redacao_text, tema)
        if verificacao["aderencia"] == "Fuga ao tema":
            return {
                "nota_final": 0,
                "competencias": {
                    "competencia_1": {"pontuacao": 0, "analise": "N/A", "pontos_fortes": "N/A", "pontos_fracos": "N/A", "sugestoes": "N/A"},
                    "competencia_2": {"pontuacao": 0, "analise": "N/A", "pontos_fortes": "N/A", "pontos_fracos": "N/A", "sugestoes": "N/A"},
                    "competencia_3": {"pontuacao": 0, "analise": "N/A", "pontos_fortes": "N/A", "pontos_fracos": "N/A", "sugestoes": "N/A"},
                    "competencia_4": {"pontuacao": 0, "analise": "N/A", "pontos_fortes": "N/A", "pontos_fracos": "N/A", "sugestoes": "N/A"},
                    "competencia_5": {"pontuacao": 0, "analise": "N/A", "pontos_fortes": "N/A", "pontos_fracos": "N/A", "sugestoes": "N/A"}
                },
                "avaliacao_geral": {
                    "avaliacao_geral": "Redação com fuga ao tema.",
                    "competencia_mais_forte": "N/A",
                    "competencia_mais_fraca": "N/A",
                    "sugestoes_prioritarias": "Revisar compreensão do tema proposto.",
                    "conclusao": "A redação apresenta fuga ao tema."
                }
            }
        
        # Avaliar cada competência
        resultados = {}
        resultados["competencia_1"] = self.avaliar_competencia_1(redacao_text)
        resultados["competencia_2"] = self.avaliar_competencia_2(redacao_text, tema)
        resultados["competencia_3"] = self.avaliar_competencia_3(redacao_text, tema)
        resultados["competencia_4"] = self.avaliar_competencia_4(redacao_text)
        resultados["competencia_5"] = self.avaliar_competencia_5(redacao_text)
        
        # Calcular nota final
        nota_final = sum([r["pontuacao"] for r in resultados.values()])
        
        # Identificar pontos fortes e fracos
        pontuacoes = [(comp, res["pontuacao"]) for comp, res in resultados.items()]
        comp_mais_forte = max(pontuacoes, key=lambda x: x[1])
        comp_mais_fraca = min(pontuacoes, key=lambda x: x[1])
        
        # Criar avaliação geral
        avaliacao_geral = self.gerar_avaliacao_geral(resultados, comp_mais_forte, comp_mais_fraca)
        
        return {
            "nota_final": nota_final,
            "competencias": resultados,
            "avaliacao_geral": avaliacao_geral
        }
    
    def avaliar_competencia_1(self, redacao_text):
        """
        Avalia o domínio da norma padrão (Competência 1)
        
        Args:
            redacao_text: Texto da redação
            
        Returns:
            Dicionário com avaliação
        """
        # Recuperar documentos sobre norma culta
        docs = self.recuperar_documentos("norma culta gramática redação enem", "norma_culta", k=2)
        contexto = "\n\n".join([doc.page_content for doc in docs])
        
        # Extrair estatísticas
        from src.utils import contar_palavras, contar_frases, contar_paragrafos
        
        estatisticas = f"""
        Estatísticas do texto:
        - Palavras: {contar_palavras(redacao_text)}
        - Frases: {contar_frases(redacao_text)}
        - Parágrafos: {contar_paragrafos(redacao_text)}
        """
        
        prompt = f"""
        Você é um especialista em correção de redações do ENEM. 
        Avalie o domínio da norma padrão (Competência 1) na redação abaixo.

        # Texto da redação:
        {redacao_text}

        # Estatísticas do texto:
        {estatisticas}

        # Contexto sobre avaliação da norma padrão no ENEM:
        {contexto}
        
        # Formato esperado da resposta
        ```json
        {{
          "pontuacao": 0-200,
          "analise": "Análise detalhada dos aspectos formais",
          "pontos_fortes": "Aspectos positivos quanto à norma padrão",
          "pontos_fracos": "Aspectos a melhorar",
          "sugestoes": "Sugestões específicas para melhorar"
        }}
        ```
        
        Responda APENAS com o JSON solicitado.
        """
        
        # Chama a API
        resultado = self.safe_api_call(prompt)
        
        # Extrair o JSON da resposta
        try:
            # Procura pelo JSON entre ```json e ```
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', resultado, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = resultado
            
            # Carrega o JSON
            resultado = json.loads(json_str)
            return resultado
        except Exception as e:
            raise Exception(f"Erro ao processar resultado da avaliação de Competência 1: {str(e)}\nResposta: {resultado}")
    
    def avaliar_competencia_2(self, redacao_text, tema):
        """
        Avalia a compreensão do tema (Competência 2)
        
        Args:
            redacao_text: Texto da redação
            tema: Tema proposto
            
        Returns:
            Dicionário com avaliação
        """
        # Recuperar documentos sobre compreensão de tema
        docs = self.recuperar_documentos("compreensão tema redação enem", "tema", k=2)
        contexto = "\n\n".join([doc.page_content for doc in docs])
        
        prompt = f"""
        Você é um especialista em correção de redações do ENEM. 
        Avalie a compreensão do tema e aplicação de conceitos (Competência 2) na redação abaixo.

        # Tema da redação:
        {tema}

        # Texto da redação:
        {redacao_text}

        # Contexto sobre avaliação da compreensão do tema no ENEM:
        {contexto}
        
        # Formato esperado da resposta
        ```json
        {{
          "pontuacao": 0-200,
          "analise": "Análise detalhada da compreensão do tema",
          "pontos_fortes": "Aspectos positivos quanto à compreensão do tema",
          "pontos_fracos": "Aspectos a melhorar",
          "sugestoes": "Sugestões específicas para melhorar"
        }}
        ```
        
        Responda APENAS com o JSON solicitado.
        """
        
        # Chama a API
        resultado = self.safe_api_call(prompt)
        
        # Extrair o JSON da resposta
        try:
            # Procura pelo JSON entre ```json e ```
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', resultado, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = resultado
            
            # Carrega o JSON
            resultado = json.loads(json_str)
            return resultado
        except Exception as e:
            raise Exception(f"Erro ao processar resultado da avaliação de Competência 2: {str(e)}\nResposta: {resultado}")
    
    def avaliar_competencia_3(self, redacao_text, tema):
        """
        Avalia a argumentação (Competência 3)
        
        Args:
            redacao_text: Texto da redação
            tema: Tema proposto
            
        Returns:
            Dicionário com avaliação
        """
        # Recuperar documentos sobre argumentação
        docs = self.recuperar_documentos("argumentação redação enem", "argumentacao", k=2)
        contexto = "\n\n".join([doc.page_content for doc in docs])
        
        # Extrair desenvolvimento para análise específica
        desenvolvimento = extrair_desenvolvimento(redacao_text)
        
        prompt = f"""
        Você é um especialista em correção de redações do ENEM. 
        Avalie a seleção e organização de argumentos (Competência 3) na redação abaixo.

        # Tema da redação:
        {tema}

        # Desenvolvimento da redação (foco da análise):
        {desenvolvimento}

        # Texto completo da redação (para contexto):
        {redacao_text}

        # Contexto sobre avaliação da argumentação no ENEM:
        {contexto}
        
        # Formato esperado da resposta
        ```json
        {{
          "pontuacao": 0-200,
          "analise": "Análise detalhada da argumentação",
          "pontos_fortes": "Aspectos positivos quanto à argumentação",
          "pontos_fracos": "Aspectos a melhorar",
          "sugestoes": "Sugestões específicas para melhorar"
        }}
        ```
        
        Responda APENAS com o JSON solicitado.
        """
        
        # Chama a API
        resultado = self.safe_api_call(prompt)
        
        # Extrair o JSON da resposta
        try:
            # Procura pelo JSON entre ```json e ```
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', resultado, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = resultado
            
            # Carrega o JSON
            resultado = json.loads(json_str)
            return resultado
        except Exception as e:
            raise Exception(f"Erro ao processar resultado da avaliação de Competência 3: {str(e)}\nResposta: {resultado}")
    
    def avaliar_competencia_4(self, redacao_text):
        """
        Avalia a coesão textual (Competência 4)
        
        Args:
            redacao_text: Texto da redação
            
        Returns:
            Dicionário com avaliação
        """
        # Recuperar documentos sobre coesão
        docs = self.recuperar_documentos("coesão textual redação enem", "coesao", k=2)
        contexto = "\n\n".join([doc.page_content for doc in docs])
        
        prompt = f"""
        Você é um especialista em correção de redações do ENEM. 
        Avalie a coesão textual (Competência 4) na redação abaixo.

        # Texto da redação:
        {redacao_text}

        # Contexto sobre avaliação da coesão no ENEM:
        {contexto}
        
        # Formato esperado da resposta
        ```json
        {{
          "pontuacao": 0-200,
          "analise": "Análise detalhada da coesão textual",
          "pontos_fortes": "Aspectos positivos quanto à coesão",
          "pontos_fracos": "Aspectos a melhorar",
          "sugestoes": "Sugestões específicas para melhorar"
        }}
        ```
        
        Responda APENAS com o JSON solicitado.
        """
        
        # Chama a API
        resultado = self.safe_api_call(prompt)
        
        # Extrair o JSON da resposta
        try:
            # Procura pelo JSON entre ```json e ```
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', resultado, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = resultado
            
            # Carrega o JSON
            resultado = json.loads(json_str)
            return resultado
        except Exception as e:
            raise Exception(f"Erro ao processar resultado da avaliação de Competência 4: {str(e)}\nResposta: {resultado}")
    
    def avaliar_competencia_5(self, redacao_text):
        """
        Avalia a proposta de intervenção (Competência 5)
        
        Args:
            redacao_text: Texto da redação
            
        Returns:
            Dicionário com avaliação
        """
        # Recuperar documentos sobre proposta de intervenção
        docs = self.recuperar_documentos("proposta intervenção redação enem", "intervencao", k=2)
        contexto = "\n\n".join([doc.page_content for doc in docs])
        
        # Extrair conclusão e proposta para análise específica
        conclusao = extrair_conclusao(redacao_text)
        proposta = extrair_proposta_intervencao(conclusao)
        
        prompt = f"""
        Você é um especialista em correção de redações do ENEM. 
        Avalie a proposta de intervenção (Competência 5) na redação abaixo.

        # Conclusão da redação (foco da análise):
        {conclusao}

        # Proposta de intervenção identificada:
        {proposta}

        # Texto completo da redação (para contexto):
        {redacao_text}

        # Contexto sobre avaliação da proposta de intervenção no ENEM:
        {contexto}
        
        # Formato esperado da resposta
        ```json
        {{
          "pontuacao": 0-200,
          "analise": "Análise detalhada da proposta de intervenção",
          "pontos_fortes": "Aspectos positivos da proposta",
          "pontos_fracos": "Aspectos a melhorar",
          "sugestoes": "Sugestões específicas para melhorar"
        }}
        ```
        
        Responda APENAS com o JSON solicitado.
        """
        
        # Chama a API
        resultado = self.safe_api_call(prompt)
        
        # Extrair o JSON da resposta
        try:
            # Procura pelo JSON entre ```json e ```
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', resultado, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = resultado
            
            # Carrega o JSON
            resultado = json.loads(json_str)
            return resultado
        except Exception as e:
            raise Exception(f"Erro ao processar resultado da avaliação de Competência 5: {str(e)}\nResposta: {resultado}")
    
    def gerar_avaliacao_geral(self, resultados, comp_mais_forte, comp_mais_fraca):
        """
        Gera uma avaliação geral da redação
        
        Args:
            resultados: Dicionário com resultados por competência
            comp_mais_forte: Tupla (competencia_id, pontuacao) da competência mais forte
            comp_mais_fraca: Tupla (competencia_id, pontuacao) da competência mais fraca
            
        Returns:
            Dicionário com avaliação geral
        """
        # Mapear nomes de competências
        nomes_comp = {
            "competencia_1": "Domínio da norma padrão",
            "competencia_2": "Compreensão do tema",
            "competencia_3": "Argumentação",
            "competencia_4": "Coesão textual",
            "competencia_5": "Proposta de intervenção"
        }
        
        # Calcular nota total
        nota_total = sum([r["pontuacao"] for r in resultados.values()])
        
        # Criar texto para competência mais forte
        comp_forte_id, comp_forte_nota = comp_mais_forte
        comp_forte_texto = f"A competência mais forte é **{nomes_comp[comp_forte_id]}** ({comp_forte_nota} pontos). {resultados[comp_forte_id]['pontos_fortes']}"
        
        # Criar texto para competência mais fraca
        comp_fraca_id, comp_fraca_nota = comp_mais_fraca
        comp_fraca_texto = f"A competência mais fraca é **{nomes_comp[comp_fraca_id]}** ({comp_fraca_nota} pontos). {resultados[comp_fraca_id]['pontos_fracos']}"
        
        # Determinar nível geral e sugestões prioritárias
        if nota_total >= 800:
            nivel = "excelente"
            conclusao = "A redação apresenta um nível excelente, com bom domínio em todas as competências. Pequenos ajustes podem aperfeiçoar ainda mais o texto."
        elif nota_total >= 600:
            nivel = "bom"
            conclusao = "A redação apresenta um bom nível, com pontos fortes claros, mas ainda há espaço para melhorias significativas em algumas competências."
        elif nota_total >= 400:
            nivel = "médio"
            conclusao = "A redação apresenta um nível médio, com aspectos positivos, mas necessita de melhorias substanciais em várias competências."
        else:
            nivel = "insuficiente"
            conclusao = "A redação apresenta um nível insuficiente, necessitando de melhorias urgentes em várias competências para alcançar um resultado satisfatório."
        
        # Sugestões prioritárias
        sugestoes = f"Priorize melhorar a competência **{nomes_comp[comp_fraca_id]}**. {resultados[comp_fraca_id]['sugestoes']}"
        
        # Avaliação geral
        avaliacao = f"A redação obteve **{nota_total}** pontos de 1000 possíveis, apresentando um nível **{nivel}**. O texto demonstra {resultados['competencia_1']['analise'].split('.')[0].lower()}, {resultados['competencia_2']['analise'].split('.')[0].lower()} e {resultados['competencia_3']['analise'].split('.')[0].lower()}."
        
        return {
            "avaliacao_geral": avaliacao,
            "competencia_mais_forte": comp_forte_texto,
            "competencia_mais_fraca": comp_fraca_texto,
            "sugestoes_prioritarias": sugestoes,
            "conclusao": conclusao
        }