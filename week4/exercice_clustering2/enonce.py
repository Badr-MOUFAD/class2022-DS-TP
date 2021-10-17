
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Manipulation du maximum de vraisemblance 2 :
#
# On veut quantifier l'efficacite d'un produit pour ameliorer le rendement d'un moteur.
#
# Les resultats quantifies sont dans le fichier 'J18_E4_QuantifiedData.csv' qui contient:
#   (1ere colonne) La quantite de produit injecte
#   (2eme colonne) Le rendement mesure
#
# Nous allons essayer de trouver la relation entre la quantite de produit injecte et le
# rendement mesure a l'aide d'un modele de regression lineaire avec differentes
# modelisations du bruit
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from scipy.stats import chi2
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt


MyData = np.genfromtxt('J18_E4_QuantifiedData.csv')

plt.scatter(MyData[:, 0], MyData[:, 1])
plt.show()


# QUESTION 1 : Essayez de resoudre le probleme a l'aide de l'algorithme de
#             regression lineaire de scikit-learn

X = MyData[:, 0]
Y = MyData[:, 1]


# TO DO


# On peut constater que la pente de la courbe est legerement trop
# forte. Cette mauvaise estimation est due a trois observations a
# droite de la figure qui font un effet levier. La regression
# lineaire minimise l'erreur d'approximation au carre sur les
# observations d'apprentissage. De maniere sous-jacente cela se
# base sur l'hypothese que les erreurs d'approximation suivent
# une loi normale centree (et pas forcement reduite). Hors, les
# erreurs d'approximation autour d'un modele lineaire sont
# clairement non symetriques ici. Nous allons alors resoudre le
# probleme au sens du maximum de vraisemblance.

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# QUESTION 2 :
#
# On va modeliser le probleme sous la forme :
#
# -> $ypred_i = a * x_i +b \,,\, \forall i = 1, \ldots, n$
# -> $err_i= ypred_i-y_i$
#
# ou les $x_i$ et $y_i$ sont les donnees d'apprentissage pour les
# observations $i$ dans $[1, 2, ..., n]$, et $ypred_i$ approche
# $y_i$. Les deux parametres du modele lineaire que l'on cherche
# a estimer sont $a$ et $b$. Afin de rÃ©soudre le probleme, on va
# alors repondre aux sous-questions suivantes :
#
# Question 2.1 : Codez une fonction qui calcul les erreurs d'approximations
#               pour toutes les observations de $X$ et $Y$ avec un $a$ et
#               un $b$ specifiques.
# Question 2.2 : Codez une fonction qui calcule la vraisemblance de
#               parametres pour lesquel l'erreur d'approximation suit une
#               loi normale centree d'ecart type sigma. On donnera la
#               valeur par defaut sigma=2
# Question 2.3 : Codez une fonction qui calcule la vraisemblance de
#               parametres pour lesquel l'erreur d'approximation suit une loi
#               de chi2. On fixera par defaut le nombre de degres de liberte
#               ddl=3 et l'echelle de la loi (scale) a 0.4. On fera trÃ¨s
#               attention au fait que la densite de probabilite d'une valeur
#               negative sera egale a zero avec la loi du chi2.
# Question 2.4 : Utilisez les fonctions de calcul de la vraisemblance pour
#               trouver une relation lineaire qui semble raisonable, i.e. pour
#               trouver les parametres a et b les plus vraisemblables.
#               On pourra eventuellement s'aider d'une representation du nuage
#               de points qui represente le 'score' attribue a chaque
#               observation.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# CODE 2.1
def compute_errors(X, Y, theta):
    """
    returns a vector having the same size as X or Y which represents the errors
    with a 1D linear model of parameters theta=[a,b]
    """

    # TO DO

    return  # errors

# CODE 2.2


def likelihood_normal(X, Y, theta, sigma=2., verbose=False):
    """
    returns the likelihood of the 1D linear model with parameters theta=[a,b] and
    the errors following a normal law of std=sigma.
    If verbose==True, the computed scores and the computed linear model will be
    represented
    """

    # TO DO

    return  # likelihood

# CODE 2.3


def likelihood_chi2(X, Y, theta, dof=3, sc=0.4, verbose=False):
    """
    returns the likelihood of the 1D linear model with parameters theta=[a,b] and
    the errors following a chi2 law of dof degrees of freedom
    If verbose==True, the computed scores and the computed linear model will be
    represented
    """
    # TO DO

    return  # likelihood

# CODE 2.4

# TO DO


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# QUESTION 3 : Codez une fonction de descente de gradient pour
#             apprendre les parametres optimaux du modele (a et b)
#             avec les deux types de bruit consideres mais leurs
#             parametres fixes aux valeurs par defaut.
#
#             Remarque: on pourra +maximiser+ la +log-vraisemblance+,
#             ce qui est numeriquement plus simple que la vraisemblance.
#
#             Une fois que cela marchera vous pourrez tenter
#             d'etendre ce travail a l'estimation jointe des
#             parametres a et b et des hyper-parametres sur
#             le bruit.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


# 3.1: fonctions pour la descente de gradient

# TO DO


# 3.2: fonctions pour calculer le maximum de vraisemblance

# TO DO
