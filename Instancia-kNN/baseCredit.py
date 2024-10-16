import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

with open('credit.pkl', 'rb') as f:
    x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste = pickle.load(f)

knn_credit = KNeighborsClassifier(n_neighbors=5, p = 2)
knn_credit.fit(x_credit_treinamento, y_credit_treinamento)

previsoes = knn_credit.predict(x_credit_teste)
print(previsoes)

print(accuracy_score(y_credit_teste, previsoes))

print(classification_report(y_credit_teste, previsoes))