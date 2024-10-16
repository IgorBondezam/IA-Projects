from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
import pickle
from sklearn import tree
import matplotlib.pyplot as plt



with open('census.pkl', 'rb') as f:
    x_census,  y_census, x_census_teste, y_census_teste= pickle.load(f)

arvores_census = DecisionTreeClassifier(criterion='entropy', random_state=0)
arvores_census.fit(x_census, y_census)

#Mostra o ganho de informação em cada atributo
print(arvores_census.feature_importances_)


previsoes = arvores_census.predict(x_census_teste, )
print(previsoes)

accuracy = accuracy_score(y_census_teste, previsoes)
print(accuracy)

print(classification_report(y_census_teste, previsoes))