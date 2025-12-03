from flask import Flask, render_template, jsonify, request
import os
import pandas as pd
import preditor

app = Flask(__name__)


# ============================
#   ROTA PRINCIPAL (DASHBOARD)
# ============================
@app.route('/')
def index():
    return render_template('index.html')



# ============================
#   ROTA DE PREDIÇÃO
# ============================
@app.route('/predict/<path:symbol>', methods=['GET'])
def predict_route(symbol):
    try:
        # Converte BTC/USDT → BTC_USDT
        safe_symbol = symbol.replace("/", "_").replace(".", "")

        result = preditor.run_prediction(safe_symbol)

        if "error" in result:
            return jsonify(result), 500

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Erro inesperado no servidor: {str(e)}"}), 500



# ============================
#   ROTA DE HISTÓRICO (MÓDULO 3)
# ============================
@app.route('/history/<path:symbol>', methods=['GET'])
def history_route(symbol):
    try:
        safe_symbol = symbol.replace("/", "_").replace(".", "")
        file_path = f"Historico_Moedas/historico_{safe_symbol}.csv"

        if not os.path.exists(file_path):
            return jsonify({"error": "Histórico não encontrado"}), 404

        df = pd.read_csv(file_path)

        # ============================
        # CÁLCULO DAS FEATURES
        # ============================

        # SMA 7 e 30
        df['sma7'] = df['close'].rolling(7).mean()
        df['sma30'] = df['close'].rolling(30).mean()

        # RSI 14
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        RS = gain / loss
        df['rsi14'] = 100 - (100 / (1 + RS))

        # ============================
        # PREPARAR JSON PARA O FRONT
        # ============================

        history = []
        for i in range(len(df)):
            history.append({
                "timestamp": df['timestamp'][i] if 'timestamp' in df else i,
                "open": float(df['open'][i]),
                "high": float(df['high'][i]),
                "low": float(df['low'][i]),
                "close": float(df['close'][i]),
                "sma7": float(df['sma7'][i]) if not pd.isna(df['sma7'][i]) else None,
                "sma30": float(df['sma30'][i]) if not pd.isna(df['sma30'][i]) else None,
                "rsi14": float(df['rsi14'][i]) if not pd.isna(df['rsi14'][i]) else None
            })

        return jsonify({
            "symbol": symbol,
            "history": history
        })

    except Exception as e:
        return jsonify({"error": f"Erro no servidor: {str(e)}"}), 500



# ============================
#   EXECUÇÃO LOCAL
# ============================
if __name__ == "__main__":
    app.run()
