from sklearn.tree import DecisionTreeClassifier
import pickle
from sklearn import tree
import matplotlib.pyplot as plt


with open('risco_credito.pkl', 'rb') as f:
    x_risco_credito, y_risco_credito= pickle.load(f)

arvores_risco_credito = DecisionTreeClassifier(criterion='entropy')
arvores_risco_credito.fit(x_risco_credito, y_risco_credito)

#Mostra o ganho de informação em cada atributo
print(arvores_risco_credito.feature_importances_)

print(tree.plot_tree(arvores_risco_credito))
previsores = ['historia', 'divida', 'garantias', 'renda']
classes = ['alto', 'baixo', 'moderado']
print(classes)
figura, eixos = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
tree.plot_tree(arvores_risco_credito, feature_names=previsores,class_names=classes, filled=True)
figura.savefig('arvore_risc_credit.png')

previsoes = arvores_risco_credito.predict([[0,0,1,2],[2,0,0,0]])
print(previsoes)