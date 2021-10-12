"""
TP5b.A : Compression d'image SVD.

"""




#1) Lecture de l'image et visualisation
import numpy as np
import matplotlib.pyplot as plt



im = plt.imread('./Brain_IRM.jpg').astype(float)


plt.imshow(im, plt.cm.gray)
plt.title('Original image')
plt.colorbar()
plt.show()


#2) Decomposition en valeurs singulieres de l'image (SVD)

[u,s,vh]=np.linalg.svd(im, full_matrices=False)

print(u.shape)
print(s.shape)
print(vh.shape)


#3) Decomposition en valeurs singulieres de l'image (SVD)


smat=np.diag(s)

im_reconstructed = np.dot(u, np.dot(smat, vh))

error=im_reconstructed-im

plt.imshow(error, plt.cm.gray)
plt.title('error')
plt.colorbar()
plt.show()



"""
Questions 1 : 
-> Quantifiez l'erreur de reconstruction entre l'image reconstruite ici et l'image reconstruite
-> Vous semble-t-elle elevee ?
-> Comparez de meme le nombre de points dans l'image aux nombre de valeurs contenues dans u,s et vh. Est-ce que cette representation de l'information est interessante en l'etat ?
"""


"""
Questions 2 : 
-> Reconstruisez maintenant l'image en utilisant seulement un quart des valeurs singulieres. 
-> Quantifiez alors le niveau d'erreur.
-> Est-il eleve?
-> Comparez a nouveau le nombre de points dans l'image aux nombre de valeurs contenues dans la version tronquee de u,s et vh. Est-ce que cette representation compressee de l'information est interessante maintenant ?
"""



"""
Questions 3 : 
-> Reconstruisez maintenant l'image en choisissant un nombre de valeurs singulieres qui permettra de preserver environ 40 pourcents de la variabilite de l'image originale. 
-> Quantifiez alors le niveau d'erreur.
-> Comparez a nouveau le nombre de points dans l'image aux nombre de valeurs contenues dans le version tronquee de u,s et vh. Conclusion ?
"""


"""
Questions 4 : 
-> Representez enfin deux courbes qui mesurent en fonction du nombre de valeurs singulieres selectionnees :
  (1) la variabilite capturee dans l'image initiale
  (2) le nombres de valeurs contenues dans le version tronquee de u,s et vh
"""
