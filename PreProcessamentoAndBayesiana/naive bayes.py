import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle


base_risco_credito = pd.read_csv('risco_credito.csv')

print(base_risco_credito)

x_risco_credito = base_risco_credito.iloc[:, 0:4].values
y_risco_credito = base_risco_credito.iloc[:, 4].values

for i in range(0, 4):
    label_encoder = LabelEncoder()
    if type(x_risco_credito[0, i]) is str:
        x_risco_credito[:, i] = label_encoder.fit_transform(x_risco_credito[:, i])

print(x_risco_credito)

with open('risco_credito.pkl', 'wb') as f:
    pickle.dump([x_risco_credito, y_risco_credito], f)


naive_risco_credito = GaussianNB()

#A função fit faz o treinamento com os valores do banco de dados
#O python já faz sozinho a correção laplaciana se necessário
naive_risco_credito.fit(x_risco_credito, y_risco_credito)

#Base para teste, exemplo 1
#Valores entre parenteses significam os valores vindo do LabelEncoder - Valores string tranformados em numeros

#Historia - boa (0) / divida - alta (0) / garantias - nenhuma (1) /  renda > - 35000 (2)

#Historia - ruim (2) / divida - alta (0) / garantias - adequada (0) /  renda < - 15000 (0)

previsao = naive_risco_credito.predict([[0,0,1,2], [2,0,0,0]])

print(previsao)

#Mostra as classes de resultado
print(naive_risco_credito.classes_)

#Faz as contagens das classes
print(naive_risco_credito.class_count_)

#valores da Probabilidade apriori
print(naive_risco_credito.class_prior_)


#-----------------Base de dados do credito---------------

with open('credit.pkl', 'rb') as f:
    x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste = pickle.load(f)

print("Criar o algoritmo com a quantidade de treinamento")
print(x_credit_treinamento.shape, y_credit_treinamento.shape)


print("Testar o algoritmo com a quantidade de Teste")
print(x_credit_teste.shape, y_credit_teste.shape)

naive_credit_data = GaussianNB()
naive_credit_data.fit(x_credit_treinamento, y_credit_treinamento)

previsao = naive_credit_data.predict(x_credit_teste)

#Temos os resultados das previsoes, agora temos que comparar com o gabarito e ver o quão confiavel é o algoritmo
print(previsao)

#Passamos o gabarito e as previsoes ele gera a porcentagem
print(accuracy_score(y_credit_teste, previsao))

#Indica os erros e os acertos
print(confusion_matrix(y_credit_teste, previsao))

#report do resultado, documentado
print(classification_report(y_credit_teste, previsao))


# -----------------------------------BASE DE DADOS DO CENSO---------------

with open('census.pkl', 'rb') as f:
    x_census_treinamento, y_census_treinamento, x_census_teste, y_census_teste = pickle.load(f)

naive_census = GaussianNB()
naive_census.fit(x_census_treinamento, y_census_treinamento)
previsoes = naive_census.predict(x_census_teste)
print(previsoes)
print(accuracy_score(y_census_teste, previsoes))

print(confusion_matrix(y_census_teste, previsoes))

print(classification_report(y_census_teste, previsoes))