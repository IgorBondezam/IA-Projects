from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn import tree
import matplotlib.pyplot as plt



with open('../census.pkl', 'rb') as f:
    x_census,  y_census, x_census_teste, y_census_teste= pickle.load(f)

random_forest_census = RandomForestClassifier(criterion='entropy', n_estimators=100, random_state=0)
random_forest_census.fit(x_census, y_census)


previsoes = random_forest_census.predict(x_census_teste, )
print(previsoes)
print(y_census_teste)

accuracy = accuracy_score(y_census_teste, previsoes)
print(accuracy)

print(classification_report(y_census_teste, previsoes))