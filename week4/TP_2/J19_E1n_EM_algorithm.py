
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Algorithme E.M. pour le melange de Gaussiennes :
#
# On observe des observations en 2D qui semblent appartenir a trois sous groupes
# differents. Nous allons utiliser un modele Gaussien pour representer la
# moyenne et le niveau d'incertitude associe a chaque groupe et un algorithme
# E.M. pour clusteriser les observations et trouver les proprietes des
# Gaussiennes.

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt

X = np.genfromtxt('J19_E1n_QuantifiedData.csv')

plt.scatter(X[:, 0], X[:, 1])
plt.show()


# QUESTION 1: Clusterisez les donnes avec un algorithme de K-means avec trois
# groupes.


# TO DO


# On constatera que les deux groupe detectes en haut de la figure semblent
# s'etendre vers le centre de la figure avec des observations peu denses.
# On se dit qu'il est alors fort possible que les trois groupes soient
# echantillonnes avec des lois gaussiennes ayant des proprietes differentes, ce
# qui conduit a avoir de mauvais resultats avec les K-means.
# Nous allons ainsi resoudre le probleme de clustering a l'aide d'un algorithme
# E.M. qui estime a la fois les clusters, et les proprietes de la distribution
# des observations dans chaque groupe.

# Pour modeliser le probleme, nous allons reproduire le modele du cours.
# Pour simplifier le probleme, on considerera que les variances des lois dans
# chaque classe seront isotropes, c'est a dire que les matrices
# Sigma_j = [ [sigma_j, 0.] , [0. , sigma_j] ] , ou sigma_j est un scalaire.
#
# Les parametres a estimer seront alors :
#  theta=(tau_1,tau_2,mu_1_x,mu_1_y,mu_2_x,mu_2_y,mu_3_x,mu_3_y,sigma_1,sigma_2,sigma_3)
# avec tau_3=1.-tau_1-tau_2 (probabilite globale d'avoir des observations dans chaque
# groupe)


# QUESTION 2: Quelle est la log-vraisemblance du modele complet (avec les X et
# les Z)? On se souviendra que
#    $log ( \Pi_{i=1}^{n} ( a_i ) )   =  \sum_{i=1}^{n} ( log (a_i) ) $
#
# Remarque : On pourra utiliser scipy.stats.multivariate_normal.pdf pour mesurer
# la densite de probabilite d'une loi normale multivariee (ici 2D).
#
# Remarque : Pour estimer la vraissemblance du modele complet, on pourra tirer
# pour l'instant au hazard (dans 0, 1, 2) les labels Z associes a chaque
# observation.


def log_likelihood(X, theta, Z):

    # TO DO

    return log_likelihood


# QUESTION 3: Testez cette fonction pour trouver manuellement une parametrisation
#            qui vous semble raisonable. La log-vraissemblance est-elle bien
#            meilleur pour une parametrisation theta qui vous semble raisonable
#            que pour d'autres qui ont peu de sens?

# TO DO


# QUESTION 4: - Faire trois figures qui montrent la probabilite estimee pour la parametrisation
#              theta que chaque observation soit dans la classe 1, 2 ou 3. On se souviendra que
#              ces trois probabilites seront utilises pour evaluer l'esperance du modele
#              optimise dans l'algorithme EM.
#            - Montrer alors la classe la plus probable pour a chaque observation pour une
#              parametrisation theta donnee
#            - Observer ces figures pour plusieurs parametrisations theta


# TO DO


# QUESTION 5: Nous allons maintenant coder la fonction d'esperance de la vraisemblance
#            dans laquelle les labels Z_i sont remplaces par leur probabilite. Cette
#            fonction sera maximisee dans l'etape "M" de l'algorithme E.M.


def funct_Q(X, theta, probabilities):

    # TO DO

    return  # Q_score

# QUESTION 6: Nous allons maintenant resoudre l'algorithme E.M. Pour y arriver, deux options
#            sont possibles lors de l'etape "M":
#            - Option 1 : Vous maximisez la fonction codee dans la question precedente avec
#                         un algorithme de descente de gradient. Si vous partez sur cette
#                         option, il faudra bien remarquer que la derivee de l'esperance de
#                         la log-vraisemblance par rapport aux \mu_k n'est pas forcement sur
#                         la meme echelle que celle par rapport aux \Sigma_k ou bien aux \tau_k.
#                         Avant d'optimiser tous ces parametres a la fois, commencez alors par
#                         n'optimiser que les parametres \mu_k.
#            - Option 2 : Vous pouvez aussi analytiquement calculer la derivee de l'esperance de
#                         la log-vraisemblance par rapport a chacun des termes a optimiser (i.e
#                         calculer son gradient) et considerer que le maximum de la fonction est
#                         obtenu pour les zeros du gradient (on a une fonction convexe).

# TO DO
