# Arquivo: app.py (VERSAO FINAL - Leve e Rapida)

from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import preditor

app = Flask(__name__)
CORS(app)

print(">>> APLICACAO WEB INICIADA (MODO PREVISAO) &lt;&lt;&lt;")
print("Esta aplicacao NAO treinara modelos. Apenas fara previsoes com modelos existentes.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict/<path:symbol>', methods=['GET'])
def predict(symbol):
    print(f"Recebida requisicao de previsao para: {symbol}")
    # Chamamos com allow_training=False por seguranca.
    # O app web NUNCA deve treinar.
    result = preditor.run_prediction(symbol, allow_training=False) 
    
    if "error" in result:
        return jsonify(result), 404 # Retorna um erro se o modelo nao existir
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
