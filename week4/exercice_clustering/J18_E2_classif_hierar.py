
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#Classification hierarchique de coordonnées géographiques
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import time

#read data
dataframe=pandas.read_csv("./J18_E2_coordinates_morroco.csv",sep=',')
print(dataframe.head)
listColNames=list(dataframe.columns)


#get usefull information
X=dataframe[['lat','lng']].values

X_with_pop=dataframe[['lat','lng','population']].values

city_names=list(dataframe['city'])

plt.scatter(X[:,1],X[:,0],c=X_with_pop[:,2],cmap='rainbow',alpha=0.5)
plt.colorbar()
plt.show()


#QUESTION 1:
# -> Allez a la page https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering
# -> Trouvez comment effectuer une classification hierarchique pour clusteriser les villes du Maroc
#    en 4 groupes a l'aide de leurs coordonnees GPS (lattitude et longitude)
# -> Que signifient les differentes options de 'linkage'?
# -> Effectuez les clusterings avec les differentes options de 'linkage'. Comparez les temps de calculs
#    et les clusters trouves
# -> Comment expliquez-vous les differences de temps et de clusterings?

from sklearn.cluster import AgglomerativeClustering

#...


#QUESTION 2:
# -> Reproduisez-ce tests avec le lien 'ward' sans definir a l'avance le nombre de
#    clusters voulus, mais plutot une distance maximum de 300km entre les elements
#    de chaque cluster. On notera que l'unite d'une coordonnee GPS correspond
#    environ a 69.47km
# -> Quel est alors le premier avantage de cette methode compare aux k-means?

#...

#QUESTION 3:
# -> On s'interesse maintenant a "J18_E2_worldcities.csv" qui contient 41001
#    villes au lieu de 162
# -> Essayez de clusteriser ces nouvelles donnees de maniere a avoir 100 clusters
#    Que constatez-vous?
# -> Allez a la page https://scikit-learn.org/stable/modules/clustering.html et
#    trouvez une methode qui passe mieux a l'echelle (scalability) pour resoudre
#    le probleme.
# -> Une fois le clustering effectue, il est quasi impossible de representer
#    toutes les donnees clusterisees. Ne representez alors qu'une ville de chaque
#    classe.

#read data
dataframe=pandas.read_csv("./J18_E2_worldcities.csv",sep=',')
print(dataframe.head)
listColNames=list(dataframe.columns)

#get usefull information
X=dataframe[['lat','lng']].values
city_names=list(dataframe['city'])

#...
