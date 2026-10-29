# Classificador de Urgência — Clínica Universitária

## O que faz

Classifica automaticamente o motivo da consulta do paciente em:

- 🟢 baixa (atendimento normal)
- 🟡 media (priorizar)
- 🔴 alta (alerta ao supervisor)

## Como funciona

- Modelo: Naive Bayes (scikit-learn)
- Entrada: texto do "Motivo da Consulta"
- Saída: nível de urgência + cor

## Como rodar

### 1. Instalar dependências

pip install flask scikit-learn pandas

### 2. Treinar o modelo

python treinar.py

### 3. Rodar o servidor

python app.py

### 4. Testar

curl -X POST http://localhost:5000/classificar \
 -H "Content-Type: application/json" \
 -d '{"motivo": "estou ansioso"}'

## Endpoints

- `GET /` → Verifica se o servidor está no ar
- `POST /classificar` → Classifica o texto

## Tecnologias

- Python 3.14
- Flask 3.1
- scikit-learn 1.9
- pandas 3.0
