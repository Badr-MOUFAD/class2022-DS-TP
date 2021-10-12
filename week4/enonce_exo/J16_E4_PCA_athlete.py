import pandas as pd

"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Les donnees athle sont constituees d'un releve des records nationaux d'athletisme de 27 pays pour 9 epreuves de course, du 100m au marathon.
Re-utiliser le protocole de l'exercice 'data_iris' pour interpreter ces donnees

En utilisant les fonctions de l'exercice 1, repondez aux questions suivantes :
- Quels sont les pays qui se distinguent sur les courtes distances ?
- Quels sont les pays qui se distinguent sur les longues distances ?
- Quels sont les pays qui se distinguent sur les distances moyennes (typiquement 800m et 1500m) ?


++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""


data_athle = pd.read_csv("data_athle.csv", header=0,
                         sep="	", decimal=",", index_col=0)
ListCountries = list(data_athle.index)
X_orig = data_athle.values

# TO DO
