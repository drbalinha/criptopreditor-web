from flask import Flask, render_template, jsonify
import preditor

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict/<path:symbol>', methods=['GET'])
def predict_route(symbol):
    try:
        result = preditor.run_prediction(symbol)
        if "error" in result:
            return jsonify(result), 500
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Erro inesperado no servidor: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
