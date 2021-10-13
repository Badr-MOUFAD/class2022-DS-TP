

"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Reduction de dimension avec l'ACP et la LDA en vue de classification de donnees en grande dimension (NOTE)
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""

from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt


# 1) chargement et pre-traitement des donnees.

# 1.1) chargement
# Chaque observation represente ici une image de chiffre manuscrit (input X) ainsi que la valeur du chiffre represente (output Y).
# Dans chaque ligne du fichier charge, la premiere valeur represente le "label" de et les 784 autres valeurs representent l'image de taille 28x28 pixels sous forme d'un vecteur.


# train=np.genfromtxt('./MNIST_data/mnist_train.csv',delimiter=',')
# test=np.genfromtxt('./MNIST_data/mnist_test.csv',delimiter=',')

train = np.genfromtxt('./mnist_train_100.csv', delimiter=',')
test = np.genfromtxt('./mnist_test_100.csv', delimiter=',')


# 1.2) traitement
X_train = train[:, 1:]+np.random.randn(100, 784)*50
y_train = train[:, 0].reshape((X_train.shape[0], 1)).astype(np.int)

X_test = test[:, 1:]
y_test = test[:, 0].reshape((X_test.shape[0], 1)).astype(np.int)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# 1.3) definition d'une fonction pour visualiser les donnees
def ShowMNISTObservation(X_data, y_data, obsNb=0):
    plt.clf()
    plt.imshow(X_data[obsNb, :].reshape((28, 28)), cmap='Greys')
    plt.title('Observation '+str(obsNb)+': Label '+str((y_data[obsNb, 0])))
    plt.colorbar()
    plt.show()


"""
Question 1 : Bien comprendre comment les donnees sont organisees et representer plusieurs images issues des donnees d'apprentissage.
-> Que represente chaque ligne de X_data ?
-> Quel est le lien entre X_data[obs,:] et y_data[obs,0] ?
"""


"""
Question 2 :
-> Quel est le nombre d'observations (n), et la dimension de chaque observation (p) ?
-> Quel est de plus le nombre de classes possibles ?
-> Pourquoi ce probleme d'apprentissage est difficile ?
"""


"""
Question 3 : A l'aide d'une methode d'arbre de decision et des donnees 'train', essayez d'apprendre a predire au mieux les labels des donnees 'test'. On utilisera sklearn.tree.DecisionTreeClassifier qui est un classifieur tres simple a interpreter.

 - Faites l'apprentissage avec DecisionTreeClassifier(max_depth=10), c'est a dire avec un arbre de decisions ayant une profondeur 10.
 - Quel pourcentage de bonnes predictions obtenez vous sur les donnees test ? On mesurera le pourcentage moyen et sa son ecart type sur 50 apprentissages car les arbres sont construits de maniere aleatoire.
 - Etes-vous satisfaits de la qualite des predictions ?
"""


"""
Question 4 : A l'aide de l'ACP, on va maintenant projeter les images dans un espace de dimension 20 qui maximise la variabilite des donnees.
- Utilisez les methodes des exercices precedents sur 'X_train' pour y arriver. Bien regarder les options de sklearn.decomposition.PCA pour ne garder que 20 composantes.
"""


"""
Question 5 : Nous allons maintenant interpreter l'estimation des valeurs et vecteurs propres.
- Affichez un plot des 20 premieres valeurs propres issues de l'ACP. Est-ce qu'utiliser 20 composantes principales vous semble raisonable pour reduire la dimension du probleme ?
- En vous inspirant de la fonction ShowMNISTObservation, representez aussi les deux premiers vecteurs propre sous forme d'image. Quels chiffres la premiere composante principale semble elle distinguer facilement ?
"""


"""
Question 6 : Une fois la base optimale apprise:
- projetez X_train sur cette base de dimension 20  -> 'X_train_projected'
- relancez a nouveau 50 fois l'apprentissage avec le classifieur de profondeur 10 et 'X_train_projected' et 'y_train'.
- Quel est maintenant le taux de bonnes predictions et sa variabilite sur la base de tests ?
- Est-ce que la reduction de dimension a eu un effet benefique ?
"""


"""
Question 7 : On va maintenant essayer d'ameliorer le resultat de la question 5 en utilisant une reduction de dimension de type "Linear Discriminant Analysis" (LDA) afin de projeter au mieux X_train, mais cette fois-ci en considerant aussi y_train.
- Que va faire la LDA en pratique ?
- Reproduisez alors le test de la question 5 avec sklearn.discriminant_analysis.LinearDiscriminantAnalysis a la place de sklearn.decomposition.PCA.
- Etes vous satisfaits du resultat ? Pouvez-vous expliquer pourquoi ?
"""


"""
Question 8 : Enfin, nous aimerions essayer d'adapter la PLS en alternative a la LDA pour essayer d'ameliorer le resultat de la question 5.
- Pourquoi utiliser la PLS n'est pas naturel ici ?
- Comment pourrait-on modifier la representation de y_train pour reduire la dimension de X_train avec une PLS ?
- Si cela vous semble possible, reproduisez alors le test de la question 5 avec sklearn.cross_decomposition.PLSRegression a la place de sklearn.decomposition.PCA et decrivez sur les resultats.
"""
