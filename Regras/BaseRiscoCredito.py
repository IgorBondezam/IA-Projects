import Orange

base_risco_credito = Orange.data.Table('risco_credito.csv')
print(base_risco_credito)

#mostrar colunas
base_risco_credito.domain

cn2 = Orange.classification.rules.CN2Learner()
regras_risco_credito = cn2(base_risco_credito)

for regras in regras_risco_credito.rule_list:
    print(regras)

previsoes = regras_risco_credito([['boa', 'alta', 'nenhuma', 'acima_35'],['ruim', 'alta', 'adequada', '0_15']])
print(previsoes)

print(base_risco_credito.domain.class_var.values)