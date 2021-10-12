
"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Les donnees 'data_iris' sont composees d'un releve de 150 observations en dimension 4 + 1.

Le but de ce TP est de se familiariser avec la pratique de l'ACP, en essayant de bien comprendre les donnees de ce jeu simple.

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""

import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt


"""
Question 1 : Recupï¿½rer les donnï¿½es de data_iris.csv avec pandas (bien faire attention au separateur et a la decimale dans le csv)
"""


"""
Question 2 : A la vue des donnes en 4d, sauriez vous facilement distinguer les differentes variï¿½tï¿½s d'Iris. Nous allons repondre a cette question en plusieurs etapes.
"""


# Question 2.1) Extrayez les donnees quantitatives dans un numpy.array et observez les. Est-ce que les differentes variables sont echantillonnees sur la meme gamme de valeurs ?

# Question 2.2)  Afin de comparer des mesures ayant la meme echelle, nous allons d'abord centrer et rï¿½duire les valeurs associees a chaque variable. Utilisez 'sklearn.preprocessing.StandardScaler' pour effectuer cette etape simplement.


# Question 2.3)    Effectuer un ACP sur les donnees centrees-reduites X. Pour cela, utilisez sklearn.decomposition.PCA


# Question 2.4) Trouvez et visualisez la variance expliquee par les composantes principales


"""
Question 3 : Combien de composantes faut-il pour expliquer plus de 94% de la variabilite entre les individus ?
"""

"""
Question 4 : Comment interpreter les axes principaux 1 et 2
"""


"""
Question 5 : En projetant les donnees sur les deux axes principaux, est-ce que vous distinguez facilement deux sous-groupes de donnees ?
"""


"""
Question 6 : Les observations [0,50] du jeu de donnees sont des Iris de la variete 'Setosa', les observations [50,100] sont de la variete 'Versicolor', et les [100,150] sont de la variï¿½tï¿½ 'Virginica'. Representez ces varietes avec un jeu de couleurs different sur les 2 composantes principales de l'ACP. Trouvez alors manuellement une regle de decision simple pour les distinguer sur ce plan principal.
"""
