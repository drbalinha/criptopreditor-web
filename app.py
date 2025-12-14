from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from functools import wraps
import os
from dotenv import load_dotenv
from preditor import get_prediction
from news_fetcher import fetch_crypto_news
from sentiment import analyze_news_sentiment
import pandas as pd
import json
from datetime import datetime
import requests

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'sua_chave_secreta_aqui')

# Usuários mockados (em produção, usar banco de dados)
USERS = {
    'admin@example.com': 'admin123',
    'user@example.com': 'user123'
}

# Taxa de câmbio (em produção, buscar em tempo real)
USD_TO_BRL = 5.44

# Moedas para análise
TOP_COINS = [
    'BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'LTC/USDT', 'BCH/USDT',
    'ADA/USDT', 'DOGE/USDT', 'DOT/USDT', 'UNI/USDT', 'LINK/USDT',
    'SOL/USDT', 'MATIC/USDT', 'ETC/USDT', 'FIL/USDT', 'AAVE/USDT',
    'MKR/USDT', 'COMP/USDT', 'SNX/USDT', 'YFI/USDT', 'AVAX/USDT'
]


def login_required(f):
    """Decorator para verificar se usuário está logado"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Rota de login"""
    if request.method == 'POST':
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        # Validar credenciais
        if email in USERS and USERS[email] == password:
            session['user'] = {'email': email}
            return jsonify({'success': True, 'message': 'Login realizado com sucesso!'})
        else:
            return jsonify({'success': False, 'message': 'Email ou senha inválidos'}), 401
    
    return render_template('login.html')


@app.route('/logout')
def logout():
    """Rota de logout"""
    session.clear()
    return redirect(url_for('login'))


@app.route('/')
@login_required
def index():
    """Dashboard principal"""
    user_email = session['user']['email']
    return render_template('index.html', user_email=user_email, usd_rate=USD_TO_BRL)


@app.route('/history')
@login_required
def history():
    """Página de histórico"""
    user_email = session['user']['email']
    return render_template('history.html', user_email=user_email, usd_rate=USD_TO_BRL)


@app.route('/api/analyze', methods=['POST'])
@login_required
def analyze():
    """API para analisar top 20 moedas com sentimento"""
    try:
        print("\n" + "="*60)
        print("🔮 INICIANDO ANÁLISE DE TOP 20 MOEDAS")
        print("="*60)
        
        results = []
        
        for i, symbol in enumerate(TOP_COINS, 1):
            print(f"\n[{i}/20] Analisando {symbol}...")
            
            try:
                # Obter predição completa
                prediction = get_prediction(symbol)
                
                if prediction and 'error' not in prediction:
                    results.append(prediction)
                    print(f"✅ {symbol} analisado com sucesso")
                else:
                    print(f"⚠️ Erro ao analisar {symbol}")
            
            except Exception as e:
                print(f"❌ Erro ao processar {symbol}: {str(e)}")
                continue
        
        print(f"\n{'='*60}")
        print(f"✅ ANÁLISE COMPLETA! {len(results)} moedas processadas")
        print(f"{'='*60}\n")
        
        return jsonify({
            'success': True,
            'data': results,
            'timestamp': datetime.now().isoformat(),
            'usd_rate': USD_TO_BRL
        })
    
    except Exception as e:
        print(f"❌ ERRO NA ANÁLISE: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/sentiment/<symbol>', methods=['GET'])
@login_required
def get_sentiment(symbol):
    """API para obter sentimento de uma moeda específica"""
    try:
        # Buscar notícias
        news_list = fetch_crypto_news(symbol, limit=10)
        
        # Analisar sentimento
        sentiment_analysis = analyze_news_sentiment(news_list)
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'sentiment': sentiment_analysis,
            'news': news_list[:5],
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        print(f"❌ Erro ao obter sentimento: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/news/<symbol>', methods=['GET'])
@login_required
def get_news(symbol):
    """API para obter notícias de uma moeda"""
    try:
        limit = request.args.get('limit', 10, type=int)
        news_list = fetch_crypto_news(symbol, limit=limit)
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'news': news_list,
            'count': len(news_list),
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        print(f"❌ Erro ao obter notícias: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/usd-rate', methods=['GET'])
@login_required
def get_usd_rate():
    """API para obter taxa USD/BRL em tempo real"""
    try:
        # Tentar obter taxa da API gratuita
        try:
            response = requests.get('https://api.exchangerate-api.com/v4/latest/USD', timeout=5)
            if response.status_code == 200:
                data = response.json()
                usd_to_brl = float(data['rates'].get('BRL', 5.44))
                
                return jsonify({
                    'success': True,
                    'usd_rate': float(usd_to_brl),
                    'timestamp': datetime.now().isoformat(),
                    'source': 'exchangerate-api.com'
                })
        except Exception as e:
            print(f"⚠️ API 1 falhou: {str(e)}")
        
        # Fallback: tentar outra API
        try:
            response = requests.get('https://api.exchangerate.host/latest?base=USD&symbols=BRL', timeout=5)
            if response.status_code == 200:
                data = response.json()
                usd_to_brl = float(data['rates'].get('BRL', 5.44))
                
                return jsonify({
                    'success': True,
                    'usd_rate': float(usd_to_brl),
                    'timestamp': datetime.now().isoformat(),
                    'source': 'exchangerate.host'
                })
        except Exception as e:
            print(f"⚠️ API 2 falhou: {str(e)}")
        
        # Fallback: tentar API do Banco Central (mais precisa)
        try:
            response = requests.get('https://api.bcb.gov.br/dados/serie/bcdata.sgs.1/dados/ultimos/1?formato=json', timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    usd_to_brl = float(data[0]['valor'])
                    
                    return jsonify({
                        'success': True,
                        'usd_rate': float(usd_to_brl),
                        'timestamp': datetime.now().isoformat(),
                        'source': 'Banco Central do Brasil'
                    })
        except Exception as e:
            print(f"⚠️ API Banco Central falhou: {str(e)}")
        
        # Se todas falharem, retornar valor padrão
        print("⚠️ Todas as APIs falharam, usando valor padrão")
        return jsonify({
            'success': True,
            'usd_rate': 5.44,
            'timestamp': datetime.now().isoformat(),
            'source': 'default'
        })
    
    except Exception as e:
        print(f"❌ Erro ao obter cotação: {str(e)}")
        return jsonify({
            'success': True,
            'usd_rate': 5.44,
            'timestamp': datetime.now().isoformat(),
            'source': 'default'
        })


@app.route('/history/<symbol>', methods=['GET'])
@login_required
def get_history(symbol):
    """API para obter histórico de uma moeda"""
    try:
        symbol_file = symbol.replace('/', '_')
        file_path = f'Historico_Moedas/historico_{symbol_file}.csv'
        
        if not os.path.exists(file_path):
            return jsonify({
                'success': False,
                'error': f'Histórico não encontrado para {symbol}'
            }), 404
        
        # Ler arquivo CSV
        df = pd.read_csv(file_path)
        
        # Converter para JSON
        history_data = {
            'symbol': symbol,
            'data': df.to_dict('records'),
            'count': len(df),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify({
            'success': True,
            'history': history_data
        })
    
    except Exception as e:
        print(f"❌ Erro ao obter histórico: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/market-sentiment', methods=['GET'])
@login_required
def get_market_sentiment():
    """API para obter sentimento geral do mercado"""
    try:
        from news_fetcher import fetch_market_news
        
        # Buscar notícias do mercado
        market_news = fetch_market_news(limit=10)
        
        # Analisar sentimento
        sentiment_analysis = analyze_news_sentiment(market_news)
        
        return jsonify({
            'success': True,
            'market_sentiment': sentiment_analysis,
            'news': market_news[:5],
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        print(f"❌ Erro ao obter sentimento do mercado: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Página não encontrada"""
    return jsonify({'error': 'Página não encontrada'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Erro interno do servidor"""
    return jsonify({'error': 'Erro interno do servidor'}), 500


if __name__ == '__main__':
    print("🚀 Iniciando Dashboard Cripto ML...")
    print(f"📊 Taxa USD/BRL: R$ {USD_TO_BRL}")
    print(f"📈 Moedas para análise: {len(TOP_COINS)}")
    print("\n🌐 Acesse: http://localhost:5000")
    print("📧 Login: admin@example.com / admin123")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
