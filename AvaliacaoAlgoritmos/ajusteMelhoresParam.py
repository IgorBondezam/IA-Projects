import pickle

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


print("-"*10 + "Arvore Decisao" + "-"*10)
parametros = {'criterion': ['gini', 'entropy'],
              'splitter': ['best', 'random'],
              'min_samples_split': [2,5,10],
              'min_samples_leaf':[1,5,10]
              }
grid_search = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid= parametros)
grid_search.fit(x_credit,y_credit)
melhores_param = grid_search.best_params_
melhor_resultado = grid_search.best_score_
print(melhores_param)
print(melhor_resultado)


print("-"*10 + "Random Florest" + "-"*10)
parametros = {'criterion': ['gini', 'entropy'],
              'n_estimators': [10,40,100,150],
              'min_samples_split': [2,5,10],
              'min_samples_leaf':[1,5,10]
              }
grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid= parametros)
grid_search.fit(x_credit,y_credit)
melhores_param = grid_search.best_params_
melhor_resultado = grid_search.best_score_
print(melhores_param)
print(melhor_resultado)

print("-"*10 + "KNN" + "-"*10)
parametros = {'n_neighbors': [3,5,10,20],
              'p': [1,2]
              }
grid_search = GridSearchCV(estimator=KNeighborsClassifier(), param_grid= parametros)
grid_search.fit(x_credit,y_credit)
melhores_param = grid_search.best_params_
melhor_resultado = grid_search.best_score_
print(melhores_param)
print(melhor_resultado)


print("-"*10 + "Regrassão Logistica" + "-"*10)
parametros = {'tol': [0.0001, 0.00001, 0.000001],
              'C': [1.0, 1.5, 2.0],
              'solver': ['lbfgs','sag','saga']
              }
grid_search = GridSearchCV(estimator=LogisticRegression(), param_grid= parametros)
grid_search.fit(x_credit,y_credit)
melhores_param = grid_search.best_params_
melhor_resultado = grid_search.best_score_
print(melhores_param)
print(melhor_resultado)

print("-"*10 + "SVM" + "-"*10)
parametros = {'tol': [0.001, 0.0001, 0.00001],
              'C': [1.0, 1.5, 2.0],
              'kernel': ['rbf','linear','poly', 'sigmoid']
              }
grid_search = GridSearchCV(estimator=SVC(), param_grid= parametros)
grid_search.fit(x_credit,y_credit)
melhores_param = grid_search.best_params_
melhor_resultado = grid_search.best_score_
print(melhores_param)
print(melhor_resultado)

print("-"*10 + "Rede Neural" + "-"*10)
parametros = {'solver': ['adam','sgb','poly', 'sigmoid'],
              'batch_size': [10,56],
              'activation': ['relu', 'logistic', 'tahn']
              }
grid_search = GridSearchCV(estimator=MLPClassifier(), param_grid= parametros)
grid_search.fit(x_credit,y_credit)
melhores_param = grid_search.best_params_
melhor_resultado = grid_search.best_score_
print(melhores_param)
print(melhor_resultado)