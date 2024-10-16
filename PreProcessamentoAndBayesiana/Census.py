import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import pickle

base_census = pd.read_csv("census.csv")
print(base_census)

grafico = px.treemap(base_census, path=['workclass'])
# grafico.show()

grafico = px.parallel_categories(base_census, dimensions=['occupation', 'relationship'])
# grafico.show()

x_census = base_census.iloc[:, 0:14].values
y_census = base_census.iloc[:, 14].values

# Convertendo dados categoricos em numericos
label_encoder_teste = LabelEncoder()
teste = label_encoder_teste.fit_transform(x_census[:, 1])
print(x_census[:, 1])
print(teste)

for i in range(0, 14):
    label_encoder = LabelEncoder()
    if type(x_census[0, i]) is str:
        x_census[:, i] = label_encoder.fit_transform(x_census[:, i])

print(x_census)

#Como esses dados podem ser vistos como algo de valor, por exemplo
#Marca de carro
#Gol Uno Tesla
# 0   1    2
#Aparenta que tem uma ordem de melhor, com isso usamos onehotencoder para transformar cada categoria em uma coluna
#   Gol Uno Testa
#1    0   1     0
#2    1    0     0
onehotencoder_census = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [1,3,5,6,7,8,9,13])], remainder='passthrough')
x_census = onehotencoder_census.fit_transform(x_census).toarray()

print(x_census)



scaler_census = StandardScaler()
x_census = scaler_census.fit_transform(x_census)


x_census_treinamento, x_census_teste, y_census_treinamento, y_census_teste = train_test_split(x_census,y_census, test_size=0.15, random_state=0)

print(x_census_treinamento.shape, y_census_treinamento.shape)
print(x_census_teste.shape, y_census_teste.shape)

# with open('census.pkl', mode='wb') as f:
#     pickle.dump([x_census_treinamento, y_census_treinamento, x_census_teste, y_census_teste], f)
