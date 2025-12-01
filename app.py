# Arquivo: app.py (VERSAO FINAL COM CORRECAO CORS)
# Este e o coracao do nosso servidor web.

from flask import Flask, jsonify, request
from flask_cors import CORS  # <--- NOVA LINHA 1: Importamos a ferramenta CORS.

# Importamos as funcoes dos nossos scripts refatorados.
from preditor import run_prediction
from sentimento_checker import get_news_sentiment

# Cria a aplicacao Flask
app = Flask(__name__)
CORS(app)  # <--- NOVA LINHA 2: Damos a permissao para a nossa aplicacao.

# Lista de moedas que o seu "ORQUESTRADOR_HIBRIDO.bat" analisava.
# Voce pode mudar esta lista a qualquer momento.
MOEDAS_PARA_ANALISAR = [
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT',
    'ADA/USDT', 'AVAX/USDT', 'DOGE/USDT', 'TRX/USDT', 'DOT/USDT',
    'LINK/USDT', 'MATIC/USDT', 'SHIB/USDT', 'LTC/USDT', 'ATOM/USDT',
    'ICP/USDT', 'ETC/USDT', 'BCH/USDT', 'UNI/USDT', 'XLM/USDT'
]

# Este endpoint principal vai simular o seu botao "Analisar TOP 20"
@app.route('/analisar_tudo', methods=['GET'])
def endpoint_analise_completa():
    print(">>> REQUISICAO RECEBIDA: Iniciando analise completa das moedas...")
    resultados_finais = []

    for moeda in MOEDAS_PARA_ANALISAR:
        print(f"--- Analisando: {moeda} ---")
        
        # 1. Executa a analise de predicao primeiro.
        resultado_predicao = run_prediction(moeda)

        # 2. Se a predicao deu erro, adiciona o erro a lista e continua.
        if "error" in resultado_predicao:
            resultados_finais.append(resultado_predicao)
            continue
            
        # 3. Se a probabilidade for alta, executa a analise de sentimento.
        if resultado_predicao["avaliacao"] == "ALTA PROBABILIDADE":
            print(f"ALTA PROBABILIDADE para {moeda}. Verificando sentimento...")
            
            # Pega so o nome da moeda (ex: 'BTC') para buscar noticias
            nome_simples_moeda = moeda.split('/')[0]
            resultado_sentimento = get_news_sentiment(nome_simples_moeda)
            
            # Adiciona o resultado do sentimento ao nosso dicionario de predicao.
            resultado_predicao.update(resultado_sentimento)
        
        # Adiciona o resultado da moeda a nossa lista final
        resultados_finais.append(resultado_predicao)

    print(">>> ANALISE COMPLETA FINALIZADA!")
    # Retorna a lista completa de resultados como uma resposta JSON.
    return jsonify(resultados_finais)

# Comando para rodar a aplicacao em modo de desenvolvimento
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
