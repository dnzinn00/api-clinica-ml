from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os

app = Flask(__name__)
CORS(app)  # Permite que o site dos seus amigos acesse a API

# Carrega o modelo e o vectorizer
with open('modelo_treinado.pkl', 'rb') as f:
    modelo = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

@app.route('/')
def home():
    return jsonify({"status": "API de Triagem no ar!"})

@app.route('/classificar', methods=['POST'])
def classificar():
    dados = request.get_json()
    if not dados or 'motivo' not in dados:
        return jsonify({'erro': 'Envie o campo "motivo"'}), 400
    
    texto = dados['motivo']
    texto_processado = vectorizer.transform([texto])
    nivel = modelo.predict(texto_processado)[0]
    
    return jsonify({
        'nivel': nivel,
        'cor': '🔴' if nivel == 'alta' else ('🟡' if nivel == 'media' else '🟢')
    })

if __name__ == '__main__':
    porta = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=porta)