import Orange

base_credito = Orange.data.Table('credit_data.csv')
print(base_credito)

#mostrar colunas
base_credito.domain

base_dividida = Orange.evaluation.testing.sample(base_credito, n=0.25)
base_treinamento = base_dividida[1]
base_teste = base_dividida[0]

cn2 = Orange.classification.rules.CN2Learner()
regras_credito = cn2(base_treinamento)

for regras in regras_credito.rule_list:
    print(regras)

previsoes = Orange.evaluation.testing.TestOnTestData(base_treinamento, base_teste, [lambda testdata: regras_credito])
print(previsoes)


print(Orange.evaluation.CA(previsoes))

print(base_credito.domain.class_var.values)