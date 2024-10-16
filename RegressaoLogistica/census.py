import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


with open('census.pkl', 'rb') as f:
    x_census_treinamento, y_census_treinamento, x_census_teste, y_census_teste = pickle.load(f)

logistic_credit = LogisticRegression(random_state= 1)
logistic_credit.fit(x_census_treinamento, y_census_treinamento)

#b0
print(logistic_credit.intercept_)
#b1, b2, b3....
print(logistic_credit.coef_)

previsoes = logistic_credit.predict(x_census_teste)

print(accuracy_score(previsoes, y_census_teste))
print(classification_report(previsoes, y_census_teste))