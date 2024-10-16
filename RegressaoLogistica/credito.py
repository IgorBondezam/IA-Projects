import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


with open('credit.pkl', 'rb') as f:
    x_credit,  y_credit, x_credit_teste, y_credit_teste= pickle.load(f)

logistic_credit = LogisticRegression(random_state= 1)
logistic_credit.fit(x_credit, y_credit)

#b0
print(logistic_credit.intercept_)
#b1, b2, b3....
print(logistic_credit.coef_)

previsoes = logistic_credit.predict(x_credit_teste)

print(accuracy_score(previsoes, y_credit_teste))
print(classification_report(previsoes, y_credit_teste))