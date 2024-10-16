import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

with open('census.pkl', 'rb') as f:
    x_census_treinamento, y_census_treinamento, x_census_teste, y_census_teste = pickle.load(f)

knn_census = KNeighborsClassifier(n_neighbors=10, p = 2)
knn_census.fit(x_census_treinamento, y_census_treinamento)

previsoes = knn_census.predict(x_census_teste)
print(previsoes)

print(accuracy_score(y_census_teste, previsoes))

print(classification_report(y_census_teste, previsoes))