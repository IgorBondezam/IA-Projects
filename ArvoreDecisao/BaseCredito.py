from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
import pickle
from sklearn import tree
import matplotlib.pyplot as plt



with open('credit.pkl', 'rb') as f:
    x_credit,  y_credit, x_credit_teste, y_credit_teste= pickle.load(f)

arvores_credit = DecisionTreeClassifier(criterion='entropy', random_state=0)
arvores_credit.fit(x_credit, y_credit)

#Mostra o ganho de informação em cada atributo
print(arvores_credit.feature_importances_)

print(tree.plot_tree(arvores_credit))
previsores = ['income', 'age', 'loan']
classes = ['0', '1']
print(classes)
figura, eixos = plt.subplots(nrows=1, ncols=1, figsize=(20,20))
tree.plot_tree(arvores_credit, feature_names=previsores,class_names=classes, filled=True)
figura.savefig('arvore_credit.png')

previsoes = arvores_credit.predict(x_credit_teste, )
print(previsoes)

accuracy = accuracy_score(y_credit_teste, previsoes)
print(accuracy)

print(classification_report(y_credit_teste, previsoes))
