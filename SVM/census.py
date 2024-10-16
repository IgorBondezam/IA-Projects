from sklearn.svm import SVC
import pickle
from sklearn.metrics import accuracy_score, classification_report



with open('census.pkl', 'rb') as f:
    x_census_treinamento, y_census_treinamento, x_census_teste, y_census_teste = pickle.load(f)

svm_credit = SVC(kernel='linear', random_state= 1, C =1.0)
svm_credit.fit(x_census_treinamento, y_census_treinamento)
previsoes = svm_credit.predict(x_census_teste)
print(accuracy_score(previsoes, y_census_teste))
print(classification_report(previsoes, y_census_teste))