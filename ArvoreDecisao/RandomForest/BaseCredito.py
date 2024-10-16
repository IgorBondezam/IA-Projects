from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle
from sklearn import tree
import matplotlib.pyplot as plt



with open('../credit.pkl', 'rb') as f:
    x_credit,  y_credit, x_credit_teste, y_credit_teste= pickle.load(f)

random_forest_credit = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)
random_forest_credit.fit(x_credit, y_credit)

previsoes = random_forest_credit.predict(x_credit_teste, )
print(previsoes)

accuracy = accuracy_score(y_credit_teste, previsoes)
print(accuracy)

print(classification_report(y_credit_teste, previsoes))
