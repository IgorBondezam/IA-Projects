import pickle

from sklearn.model_selection import cross_val_score, KFold

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

with open('credit.pkl', 'rb') as f:
    x_credit,  y_credit, x_credit_teste, y_credit_teste= pickle.load(f)

x_credit = np.concatenate((x_credit, x_credit_teste), axis = 0)
print(x_credit.shape)
y_credit = np.concatenate((y_credit, y_credit_teste), axis = 0)
print(y_credit.shape)

resultado_arvore = []
resultado_random_forest = []
resultado_knn = []
resultado_logistic = []
resultado_svm = []
resultado_rede_neural = []

for i in range(30):
    kfold = KFold(n_splits=10, shuffle=True, random_state=i)

    arvore = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=1, min_samples_split=5, splitter='best')
    scores = cross_val_score(arvore, x_credit, y_credit, cv=kfold)
    #print(scores)
    #print(scores.mean())
    resultado_arvore.append(scores.mean())

    random_forest = RandomForestClassifier(criterion='entropy', min_samples_leaf=1, min_samples_split=5, n_estimators=10)
    scores = cross_val_score(random_forest, x_credit, y_credit, cv=kfold)
    # print(scores)
    # print(scores.mean())
    resultado_random_forest.append(scores.mean())

    knn = KNeighborsClassifier()
    scores = cross_val_score(knn, x_credit, y_credit, cv=kfold)
    # print(scores)
    # print(scores.mean())
    resultado_knn.append(scores.mean())

    logistic = LogisticRegression(C=1.0, solver='lbfgs', tol=0.0001)
    scores = cross_val_score(logistic, x_credit, y_credit, cv=kfold)
    # print(scores)
    # print(scores.mean())
    resultado_logistic.append(scores.mean())

    svm = SVC(kernel='rbf',C=2.0)
    scores = cross_val_score(svm, x_credit, y_credit, cv=kfold)
    # print(scores)
    # print(scores.mean())
    resultado_svm.append(scores.mean())

    rede_neural = MLPClassifier(activation='relu', batch_size=56, solver='adam')
    scores = cross_val_score(rede_neural, x_credit, y_credit, cv=kfold)
    # print(scores)
    # print(scores.mean())
    resultado_rede_neural.append(scores.mean())

print(resultado_arvore)
print(resultado_random_forest)
print(resultado_knn)
print(resultado_logistic)
print(resultado_svm)
print(resultado_rede_neural)