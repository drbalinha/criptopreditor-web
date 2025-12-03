from flask import Flask, render_template, jsonify
import preditor

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict/<path:symbol>', methods=['GET'])
def predict_route(symbol):
    try:
        # Converte BTC/USDT → BTC_USDT para casar com os arquivos CSV
        safe_symbol = symbol.replace("/", "_").replace(".", "")
        
        result = preditor.run_prediction(safe_symbol)
        
        if "error" in result:
            return jsonify(result), 500

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Erro inesperado no servidor: {str(e)}"}), 500

if __name__ == "__main__":
    app.run()
