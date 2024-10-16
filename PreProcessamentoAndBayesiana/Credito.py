import pickle

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

base_credit = pd.read_csv('credit_data.csv')
pio.renderers.default = 'svg'

# Visualiza os 8 primeiros
print(base_credit.head(8))
print(15*"-")
# Visualiza os 8 ultimos
print(base_credit.tail(8))
print(15*"-")
# Visualiza descrições como contagem, média, desvio padrao, minimo, max
print(base_credit.describe())

print(15*"-")
# Buscando dado especifico baseado no ganho
# Essa busca foi feita procurando as informações da pessoa com maior ganho
print(base_credit[base_credit['income'] >= 69995.685578])

print(15*"-")
# Mostra os valores possíveis e com o return_counts a contagem respectivo de cada valor
print(np.unique(base_credit['default'], return_counts=True))

# Cria um grafico de colunas sobre a quantidade de pessoas olhando a coluna default
print(sns.countplot(x=base_credit['default']))
plt.show()

# Cria um grafico de colunas sobre a idade de pessoas
print(plt.hist(x = base_credit['age']));
plt.show()

# Cria um grafico de colunas sobre o salario de pessoas
print(plt.hist(x = base_credit['income']));
plt.show()

# Cria um grafico de colunas sobre a divida de pessoas
print(plt.hist(x = base_credit['loan']));
plt.show()

# grafico x e y (scatter)
grafico = px.scatter_matrix(base_credit, dimensions=['age', 'income', 'loan'], color='default')
# grafico.show()

# loc é usado para localizar em uma base alguma expressao
print(base_credit.loc[base_credit['age'] < 0])


# Excluir uma coluna INTEIRA
base_credit2 = base_credit.drop('age', axis=1)
print(base_credit2)

#Excluir linhas com valores inconsistentes - Tirando do index
base_credit3 = base_credit.drop(base_credit[base_credit['age'] < 0].index)
print(base_credit3)

#Editar/preencher os dados manualmente
#.mean mostra a media
print(base_credit.mean())
print(base_credit['age'].mean())
print(base_credit['age'][base_credit['age'] > 0].mean())
base_credit.loc[base_credit['age'] < 0] = 40.92
print(base_credit['age'].head(26))


# tratamento de dados

#Ve quais valores são nulos ou n
print(base_credit.isnull())

#Faz o somatorio de cada coluna vendo nesse caso quem é nulo
print(base_credit.isnull().sum())

#Mostra quem a idade é nulo
print(base_credit.loc[pd.isnull(base_credit['age'])])

#Substitui o valor da idade que for nulo pela media
#inplace é para alterar na tabela, não só na memória
base_credit['age'].fillna(base_credit['age'].mean(), inplace= True)

#Mostra os valores que tem id = 29 ou ( | ) id = 31
print(base_credit.loc[(base_credit['clientid'] == 29) |
                (base_credit['clientid'] == 31)])

#Mesmo código de cima mas validando um conjunto de ids
print(base_credit.loc[base_credit['clientid'].isin([29,31,32])])


#Separar previsores e classes

#Vamos colocar no x os previsores
#Usando iloc separamos[quantidade de linhas, quais colunas]
#Nesse caso estamos pegando as colunas 1,2,3 e o values convertando para o tipo numpy
x_credit = base_credit.iloc[:, 1:4].values
print(x_credit)
#Nesse caso estamos pegando a coluna 4 e o values convertando para o tipo numpy
y_credit = base_credit.iloc[:, 4].values
print(y_credit)


print(x_credit[:,0].min(), x_credit[:,1].min(), x_credit[:,2].min())

print(x_credit[:,0].max(), x_credit[:,1].max(), x_credit[:,2].max())

scaler_credit = StandardScaler()
#Faz a padronização dos dados,não precisando usar a forma de padronização
x_credit = scaler_credit.fit_transform(x_credit)
print(x_credit[:,0].min(), x_credit[:,1].min(), x_credit[:,2].min())
print(x_credit[:,0].max(), x_credit[:,1].max(), x_credit[:,2].max())

print(x_credit)


x_credit_treinamento, x_credit_teste, y_credit_treinamento, y_credit_teste = train_test_split(x_credit,y_credit, test_size=0.25, random_state=0)

print(x_credit_treinamento.shape)

# with open('credit.pkl', mode='wb') as f:
#     pickle.dump([x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste], f)