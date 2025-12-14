import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestRegressor
import joblib
import time
from news_fetcher import fetch_crypto_news
from sentiment import analyze_news_sentiment
import traceback # Importar para depuração de erros

# Caminho para armazenar os modelos treinados
MODEL_STORAGE_PATH = 'temp_models'
os.makedirs(MODEL_STORAGE_PATH, exist_ok=True)

# Cache para DataFrames (opcional, pode ser removido se não for usado)
df_cache = {}


def calculate_rsi(series, period=14):
    """Calcula o RSI (Relative Strength Index) para uma série de preços."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def create_features(df):
    """
    Cria features técnicas avançadas a partir de um DataFrame de OHLCV.
    Inclui lags, médias móveis, volatilidade, RSI, MACD, Bollinger Bands,
    features de volume, momentum, ROC, ATR, Stochastic Oscillator, OBV e crossovers.
    """
    df = df.copy()
    
    # ========== FEATURES ORIGINAIS ==========
    
    # Lags (preços passados)
    for i in range(1, 8):
        df[f'close_lag_{i}'] = df['close'].shift(i)
    
    # Médias móveis simples
    df['sma7'] = df['close'].rolling(7).mean()
    df['sma30'] = df['close'].rolling(30).mean()
    df['sma90'] = df['close'].rolling(90).mean()
    
    # Volatilidade básica
    df['volatility'] = df['high'] - df['low']
    
    # RSI (Relative Strength Index)
    df['rsi14'] = calculate_rsi(df['close'], 14)
    
    # ========== FEATURES AVANÇADAS (NÍVEL 2) ==========
    
    # 1. MACD (Moving Average Convergence Divergence)
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']
    
    # 2. Bollinger Bands
    df['bb_middle'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # 3. Volume Features
    df['volume_sma7'] = df['volume'].rolling(7).mean()
    df['volume_sma30'] = df['volume'].rolling(30).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma7']
    
    # 4. Momentum
    df['momentum_5'] = df['close'] - df['close'].shift(5)
    df['momentum_10'] = df['close'] - df['close'].shift(10)
    df['momentum_20'] = df['close'] - df['close'].shift(20)
    
    # 5. Rate of Change (ROC)
    df['roc_5'] = ((df['close'] - df['close'].shift(5)) / df['close'].shift(5)) * 100
    df['roc_10'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100
    
    # 6. ATR (Average True Range) - Volatilidade real
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift(1))
    df['tr3'] = abs(df['low'] - df['close'].shift(1))
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr'] = df['true_range'].rolling(14).mean()
    
    # 7. Stochastic Oscillator
    low_14 = df['low'].rolling(14).min()
    high_14 = df['high'].rolling(14).max()
    df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    
    # 8. OBV (On-Balance Volume)
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv_sma'] = df['obv'].rolling(20).mean()
    
    # 9. Price Position (onde está em relação aos extremos)
    df['price_position'] = (df['close'] - df['low'].rolling(14).min()) / (df['high'].rolling(14).max() - df['low'].rolling(14).min())
    
    # 10. Crossovers (cruzamentos de médias)
    df['sma_cross'] = (df['sma7'] > df['sma30']).astype(int)
    df['ema_cross'] = (df['ema12'] > df['ema26']).astype(int)
    
    # Limpar colunas auxiliares usadas para cálculo de features
    df = df.drop(['tr1', 'tr2', 'tr3', 'true_range', 'ema12', 'ema26', 'bb_middle', 'bb_std'], axis=1, errors='ignore')
    
    return df


def load_models(symbol, df, sentiment_score=50):
    """
    Carrega modelos de ML pré-treinados ou os treina se não existirem.
    Usa XGBoost para classificação de direção e RandomForest para regressão de preço.
    """
    
    # Escapar o símbolo para usar como nome de arquivo (ex: BTC/USDT -> BTC_USDT)
    safe_symbol = symbol.replace('/', '_')
    
    model_path = f"{MODEL_STORAGE_PATH}/model_xgb_{safe_symbol}.pkl"
    reg_path = f"{MODEL_STORAGE_PATH}/reg_rf_{safe_symbol}.pkl"
    
    print(f"\n{'='*60}")
    print(f"🧠 TREINANDO MODELO: {symbol}")
    print(f"📊 Sentimento do Mercado: {sentiment_score}/100")
    print(f"{'='*60}")
    
    # Preparar dados para classificador e regressor
    dfX = create_features(df.copy())
    dfX["target"] = (dfX["close"].shift(-1) > dfX["close"]).astype(int) # 1 se o preço subir, 0 se descer
    dfX = dfX.dropna() # Remover linhas com valores NaN após a criação das features
    
    print(f"📊 Total de linhas disponíveis: {len(df)}")
    print(f"📊 Linhas após features e dropna: {len(dfX)}")
    
    # Seleção de features a serem usadas pelos modelos
    features = [c for c in dfX.columns if any(x in c for x in [
        'lag', 'sma', 'rsi', 'volatility',
        'macd', 'bb_', 'volume_', 'momentum',
        'roc', 'atr', 'stoch', 'obv', 'price_position', 'cross'
    ])]
    
    print(f"📊 Total de features: {len(features)}")
    
    # XGBoost Classifier (para prever a direção do preço)
    print(f"\n🔄 Treinando XGBoost Classifier...")
    classifier = XGBClassifier(
        n_estimators=150,       # Número de árvores
        max_depth=8,            # Profundidade máxima de cada árvore
        learning_rate=0.05,     # Taxa de aprendizado
        subsample=0.8,          # Fração de amostras usadas para treinar cada árvore
        colsample_bytree=0.8,   # Fração de features usadas para treinar cada árvore
        random_state=42,        # Semente para reprodutibilidade
        verbosity=0             # Suprimir mensagens de saída do XGBoost
    )
    classifier.fit(dfX[features], dfX["target"])
    print(f"✅ Classifier treinado")
    
    # Random Forest Regressor (para prever o preço exato)
    print(f"🔄 Treinando Random Forest Regressor...")
    regressor = RandomForestRegressor(
        n_estimators=200,       # Número de árvores na floresta
        max_depth=15,           # Profundidade máxima de cada árvore
        random_state=42,        # Semente para reprodutibilidade
        n_jobs=-1               # Usar todos os núcleos da CPU disponíveis
    )
    regressor.fit(dfX[features], dfX["close"])
    print(f"✅ Regressor treinado")
    
    # Salvar os modelos treinados em disco
    joblib.dump(classifier, model_path)
    joblib.dump(regressor, reg_path)
    
    print(f"💾 Modelos salvos")
    
    return classifier, regressor, features


def predict_price(symbol, df, sentiment_score=50):
    """
    Faz a predição da direção e do preço futuro de uma criptomoeda,
    ajustando o preço com base no sentimento de notícias.
    """
    
    try:
        # Escapar o símbolo para usar como nome de arquivo
        safe_symbol = symbol.replace('/', '_')
        
        model_path = f"{MODEL_STORAGE_PATH}/model_xgb_{safe_symbol}.pkl"
        reg_path = f"{MODEL_STORAGE_PATH}/reg_rf_{safe_symbol}.pkl"
        
        # Carregar modelos se existirem, senão treiná-los
        if not os.path.exists(model_path) or not os.path.exists(reg_path):
            classifier, regressor, features = load_models(symbol, df, sentiment_score)
        else:
            classifier = joblib.load(model_path)
            regressor = joblib.load(reg_path)
            # Recriar features para garantir que a lista 'features' esteja correta
            dfX = create_features(df.copy())
            features = [c for c in dfX.columns if any(x in c for x in [
                'lag', 'sma', 'rsi', 'volatility',
                'macd', 'bb_', 'volume_', 'momentum',
                'roc', 'atr', 'stoch', 'obv', 'price_position', 'cross'
            ])]
        
        # Preparar os dados mais recentes para a predição
        dfX = create_features(df.copy())
        dfX = dfX.dropna()
        
        if len(dfX) == 0:
            print(f"⚠️ Não há dados suficientes para predição após criar features para {symbol}.")
            return None
        
        # Pegar a última linha do DataFrame para fazer a predição
        X_last = dfX[features].iloc[-1:].values
        
        # Fazer as predições
        direction_prob = classifier.predict_proba(X_last)[0] # Probabilidade de subir/descer
        predicted_price = regressor.predict(X_last)[0]      # Preço previsto
        
        # Ajustar a predição de preço com base no sentimento de notícias
        # O fator de sentimento varia de -0.5 a 0.5. O ajuste é de até 2% do preço previsto.
        sentiment_factor = (sentiment_score - 50) / 100  # Normaliza para -0.5 a 0.5
        price_adjustment = predicted_price * sentiment_factor * 0.02 # Ajuste de até 2%
        predicted_price = predicted_price + price_adjustment
        
        # Determinar a direção e a confiança da predição
        confidence = max(direction_prob) * 100
        direction = "ALTA" if direction_prob[1] > 0.5 else "BAIXA"
        
        return {
            'predicted_price': float(predicted_price), # Converter para float nativo
            'direction': str(direction),               # Converter para string nativa
            'confidence': float(confidence),           # Converter para float nativo
            'sentiment_adjusted': True
        }
    
    except Exception as e:
        print(f"❌ Erro na predição de {symbol}: {str(e)}")
        traceback.print_exc() # Imprimir o stack trace completo para depuração
        return None


def predict_multi_horizon(symbol, df, sentiment_score=50):
    """
    Gera predições de preço para múltiplos horizontes de tempo (1, 3, 5, 7 dias),
    ajustando-as com base no sentimento de notícias.
    """
    
    try:
        predictions = {}
        current_price = df['close'].iloc[-1]
        
        # Obter a predição base de 1 dia
        base_pred = predict_price(symbol, df, sentiment_score)
        
        if base_pred is None:
            return None
        
        # Multiplicador de sentimento: amplifica ganhos/perdas com base no sentimento
        # Varia de 0.9 (sentimento muito bearish) a 1.1 (sentimento muito bullish)
        sentiment_multiplier = 1 + (sentiment_score - 50) / 500
        
        # Calcular predições para diferentes horizontes de dias
        # A lógica aqui é uma extrapolação simplificada da predição de 1 dia
        predictions['1'] = current_price * (1 + (base_pred['predicted_price'] - current_price) / current_price * 1.0 * sentiment_multiplier)
        predictions['3'] = current_price * (1 + (base_pred['predicted_price'] - current_price) / current_price * 1.5 * sentiment_multiplier)
        predictions['5'] = current_price * (1 + (base_pred['predicted_price'] - current_price) / current_price * 2.0 * sentiment_multiplier)
        predictions['7'] = current_price * (1 + (base_pred['predicted_price'] - current_price) / current_price * 2.5 * sentiment_multiplier)
        
        return predictions
    
    except Exception as e:
        print(f"❌ Erro nas predições multi-horizon de {symbol}: {str(e)}")
        traceback.print_exc()
        return None


def get_prediction(symbol):
    """
    Função principal que orquestra a busca de dados, análise de sentimento,
    treinamento/carregamento de modelos e geração de predições completas para uma criptomoeda.
    Retorna um dicionário com todas as informações necessárias para o frontend.
    """
    
    try:
        print(f"\n🔍 Processando {symbol}...")
        
        # Importar ccxt aqui para evitar problemas de importação circular ou atraso
        import ccxt
        exchange = ccxt.binance() # Usando Binance como exemplo
        
        # Buscar dados históricos OHLCV (Open, High, Low, Close, Volume)
        try:
            # Limite de 500 velas de 1 hora
            ohlcv = exchange.fetch_ohlcv(symbol, '1h', limit=500)
        except Exception as e:
            print(f"⚠️ Erro ao buscar dados para {symbol}: {str(e)}")
            return {'error': f'Não foi possível buscar dados para {symbol} na exchange.'}
        
        # Converter dados OHLCV para DataFrame do Pandas
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Preço atual (último preço de fechamento), convertido para float nativo
        current_price = float(df['close'].iloc[-1])
        
        # ========== BUSCAR NOTÍCIAS E SENTIMENTO ==========
        print(f"📰 Buscando notícias para {symbol}...")
        coin_symbol = symbol.split('/')[0] # Extrai o nome da moeda (ex: BTC de BTC/USDT)
        news_list = fetch_crypto_news(coin_symbol, limit=5) # Busca as 5 notícias mais recentes
        
        # Analisar o sentimento das notícias
        sentiment_analysis = analyze_news_sentiment(news_list)
        sentiment_score = int(sentiment_analysis['average_score']) # Score médio, convertido para int nativo
        sentiment_direction = str(sentiment_analysis['average_sentiment']) # Direção, convertido para string nativa
        
        print(f"📊 Sentimento: {sentiment_direction} ({sentiment_score}/100)")
        
        # ========== FAZER PREDIÇÕES ==========
        print(f"🔮 Fazendo predições...")
        
        # Predição base (1 dia)
        base_pred = predict_price(symbol, df, sentiment_score)
        
        if base_pred is None:
            return {'error': f'Não foi possível fazer predição de preço base para {symbol}'}
        
        # Predições para múltiplos horizontes de tempo
        multi_pred = predict_multi_horizon(symbol, df, sentiment_score)
        
        if multi_pred is None:
            return {'error': f'Não foi possível fazer predições multi-horizon para {symbol}'}
        
        # Montar o dicionário de resultados, garantindo que todos os tipos sejam nativos do Python
        return {
            'symbol': str(symbol),
            'current_price': float(round(current_price, 2)),
            'prediction_direction': str(base_pred['direction']),
            'prediction_confidence': float(round(base_pred['confidence'], 2)),
            'predicted_price_1': float(round(base_pred['predicted_price'], 2)),
            'horizons': [
                float(round(multi_pred['1'], 2)),
                float(round(multi_pred['3'], 2)),
                float(round(multi_pred['5'], 2)),
                float(round(multi_pred['7'], 2))
            ],
            'sentiment': {
                'direction': str(sentiment_direction),
                'score': int(sentiment_score),
                'bullish_count': int(sentiment_analysis['bullish_count']),
                'bearish_count': int(sentiment_analysis['bearish_count']),
                'neutral_count': int(sentiment_analysis['neutral_count'])
            },
            'news': [
                {
                    'title': str(news.get('title', 'Sem título')),
                    'sentiment': str(news.get('sentiment', 'NEUTRO')),
                    'score': int(news.get('score', 50)),
                    'source': str(news.get('source', 'Desconhecido'))
                }
                for news in sentiment_analysis['details'][:3] # Limita a 3 notícias para o frontend
            ]
        }
    
    except Exception as e:
        print(f"❌ Erro geral ao processar {symbol}: {str(e)}")
        traceback.print_exc() # Imprimir o stack trace completo para depuração
        return {'error': str(e)}
