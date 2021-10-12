
"""
J17_E2 : sparse PLS.

Nous allons maintenant creuser l'algorithme de la PLS avec des donnees qui contiennent des
relations plus complexes que dans l'exercice precedent. Nous allons aussi essayer de
detecter automatiquement les variables les plus pertinentes a l'aide d'une
strategie L1 de parcimonie.
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import scale

# Generation de donnees
n = 1000
p = 10
X = np.random.normal(size=n * p).reshape((n, p))
X[:, 2] = 0.5*X[:, 1]+0.1*np.random.normal(size=n)
X[:, 3] = 0.4*X[:, 1]+0.1*np.random.normal(size=n)
y = 1.5 * X[:, 0] + 1 * X[:, 1] + np.random.normal(size=n)


def PLS1(X, y, l):
    """
    PLS1 strategy - NIPALS algorithm

    Returns:
        - B: regression coefficients wrt the original predictors
        - W: orthonormal weights
        - P: matrix of X-loading
        - q: vector of y-loading
    """
    [n, p] = X.shape

    list_w_k = []
    list_p_k = []
    list_q_k = []

    X_k = X.copy()
    v_k = np.dot(X_k.transpose(), y)
    w_k = v_k/np.linalg.norm(v_k)

    for k in range(l):
        # compute the projections and loadings
        tau_k = np.dot(X_k, w_k)
        t_k = tau_k/np.linalg.norm(tau_k)
        p_k = np.dot(X_k.transpose(), t_k)
        q_k = np.dot(t_k.transpose(), y)

        # save the elements for the matrices
        list_w_k.append(w_k)
        list_p_k.append(p_k)
        list_q_k.append(q_k)

        # prepare the next iteration
        if k < l-1:
            X_k = X_k-np.dot(t_k.reshape(-1, 1), p_k.reshape(1, -1))
            v_k = np.dot(X_k.transpose(), y)
            w_k = v_k/np.linalg.norm(v_k)

    # create the martices
    W = np.zeros([p, l])
    P = np.zeros([p, l])
    q = np.zeros(l)
    for i in range(l):
        W[:, i] = list_w_k[i]
        P[:, i] = list_p_k[i]
        q[i] = list_q_k[i]

    tmp = np.linalg.inv(np.dot(P.transpose(), W))
    B = np.dot(np.dot(W, tmp), q)

    return [B, W, P, q]


# Data normalization
X = scale(X)
y = scale(y)

[B, W, P, q] = PLS1(X, y, 3)

plt.plot(W[:, 0], 'b')
plt.plot(W[:, 1], 'g')
plt.plot(W[:, 2], 'r')
plt.title(str(np.round(q[0], 3))+' ' +
          str(np.round(q[1], 3))+' '+str(np.round(q[2], 3)))
plt.show()

# +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++
# Question 1: Bien comprendre le detail de l'algorithme NIPALS qui est dans la fonction PLS1
#            Que representent les courbes du plot et les valeurs de q ?
# +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++


# +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++
# Question 2: Dans les lignes ci-dessous, quel est le lien entre X, T et X_reco ?
# +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++

T = np.dot(X, W)
X_reco = np.dot(T, W.transpose())

for i in range(3):
    print("obs "+str(i)+":")
    print(X[i, :])
    print(X_reco[i, :])


# +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++
# Question 3: Que fait on dans les lignes ci-dessous ?
# +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++

Y_pred = np.dot(X, np.vstack(B))
plt.plot(Y_pred, y, '.')
plt.show()


# +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++
# Question 4: En vous inspirant du cours, modifiez la fonction PLS1 de maniere a avoir
#            une base de projecteurs parcimonieuse. Utilisez la pour trouver une base
#            plus simple a expliquer que precedement.
# +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++  +++


def sPLS1(X, y, l, lambda_val):
    """
    sparse PLS1 strategy - NIPALS algorithm

    Returns:
        - B: regression coefficients wrt the original predictors
        - W: orthonormal weights
        - P: matrix of X-loading
        - q: vector of y-loading
    """

    # TO DO

    return [B, W, P, q]


lambda_val = 0  # TO DEFINE

[B, W, P, q] = sPLS1(X, y, 3, lambda_val)

plt.plot(W[:, 0], 'b')
plt.plot(W[:, 1], 'g')
plt.plot(W[:, 2], 'r')
plt.title(str(np.round(q[0], 3))+' ' +
          str(np.round(q[1], 3))+' '+str(np.round(q[2], 3)))
plt.show()


T_sparse = np.dot(X, W)
X_reco_sparse = np.dot(T_sparse, W.transpose())

print(np.abs(X[:, 0:4]).std())
print(np.abs(X[:, 0:4]-X_reco[:, 0:4]).std())
print(np.abs(X[:, 0:4]-X_reco_sparse[:, 0:4]).std())
