import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

dados = pd.DataFrame({
    'texto': [
        'estou triste mas quero melhorar',
        'pensando em me machucar',
        'ansiedade antes das provas',
        'quero desistir de tudo',
        'me sinto sozinho',
        'tenho pensado em suicídio',
        'não consigo dormir',
        'estou com medo do futuro',
        'quero aprender a lidar com estresse',
        'sinto que não tenho mais saída'
    ],
    'nivel': [
        'baixa', 'alta', 'media', 'alta',
        'baixa', 'alta', 'media',
        'baixa', 'baixa', 'alta'
    ]
})

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(dados['texto'])
modelo = MultinomialNB()
modelo.fit(X, dados['nivel'])

# 3. Salva o modelo treinado (para usar depois)
with open('modelo_treinado.pkl', 'wb') as f:
    pickle.dump(modelo, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("✅ Modelo treinado e salvo!")