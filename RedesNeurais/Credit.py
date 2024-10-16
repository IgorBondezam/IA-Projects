import pickle
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier

with open('credit.pkl', 'rb') as f:
    x_credit,  y_credit, x_credit_teste, y_credit_teste= pickle.load(f)


# max_iter = quantidade de dados dados q serão usados para o teste
# verbose = mostrar uma frase ajudando a definir o minimo total
# tol = nivel de tolerancia q ele irá parar, mesmo se não chegou ao valor maximo da iteração
# activation = algoritmo q será usado para se ter a classe de saida(sigmoid, linear....)
# solver = tipo de gradiente de descida - stochastic, batch .....
# numeros de camadas ocultas e quantidade de neuronios das camadas ocultas
rede_neural_credit = MLPClassifier(max_iter=1500, verbose=True, tol=0.0000100,
                                   solver= 'adam', activation= 'relu',
                                   hidden_layer_sizes= (2,2))
                                                #2 camadas com 2 neuronios cada camada oculta
rede_neural_credit.fit(x_credit, y_credit)

previsoes = rede_neural_credit.predict(x_credit_teste)
print(previsoes)
print(accuracy_score(y_credit_teste, previsoes))
print(classification_report(y_credit_teste, previsoes))