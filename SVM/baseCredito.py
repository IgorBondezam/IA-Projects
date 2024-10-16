from sklearn.svm import SVC
import pickle
from sklearn.metrics import accuracy_score, classification_report



with open('credit.pkl', 'rb') as f:
    x_credit,  y_credit, x_credit_teste, y_credit_teste= pickle.load(f)

svm_credit = SVC(kernel='rbf', random_state= 1, C =2.0)
svm_credit.fit(x_credit, y_credit)
previsoes = svm_credit.predict(x_credit_teste)
print(accuracy_score(previsoes, y_credit_teste))
print(classification_report(previsoes, y_credit_teste))