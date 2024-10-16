import pickle
from sklearn.linear_model import LogisticRegression
import numpy as np

with open('risco_credito.pkl', 'rb') as f:
    x_risco_credito, y_risco_credito= pickle.load(f)

# retirar os valores que possuem moderado, deixando só alto e baixo
x_risco_credito = np.delete(x_risco_credito, [2, 7, 11], axis= 0)
y_risco_credito = np.delete(y_risco_credito, [2, 7, 11], axis= 0)

logistic_risco_credit = LogisticRegression(random_state= 1)
logistic_risco_credit.fit(x_risco_credito, y_risco_credito)

print(logistic_risco_credit.intercept_)
print(logistic_risco_credit.coef_)

previsoes = logistic_risco_credit.predict([[0,0,1,2], [2,0,0,0]])
print(previsoes)