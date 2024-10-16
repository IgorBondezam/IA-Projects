import pickle
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier

with open('census.pkl', 'rb') as f:
    x_census_treinamento, y_census_treinamento, x_census_teste, y_census_teste = pickle.load(f)


rede_neural_census = MLPClassifier(verbose=True, max_iter=1000, tol=0.000010,
                                   hidden_layer_sizes=(55, 55))
rede_neural_census.fit(x_census_treinamento, y_census_treinamento)

previsoes = rede_neural_census.predict(x_census_teste)

print(accuracy_score(y_census_teste, previsoes))

print(classification_report(y_census_teste, previsoes))