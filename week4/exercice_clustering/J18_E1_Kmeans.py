
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#TP1_Exo2) segmentation d'image couleur avec l'algorithme des K-means
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np                 #pour faire des mathematiques numeriques
import matplotlib.pyplot as plt    #pour afficher des images (fonction imshow)
import numpy.random as nprand      #pour tirer des valeurs aléatoirement
import scipy.ndimage as scim       #pour tout ce qui tourne autour du filtrage et le traitement d'image de base
import matplotlib.pyplot

#-----------------------------------------------------------------------
#1) fonctions de visualisation des images
#-----------------------------------------------------------------------

def showImage3channels(InputImage):
  plt.figure(1)
  plt.imshow(InputImage[:,:,0],cmap='Greys')
  plt.title('channel 1: red')
  plt.colorbar()
  
  plt.figure(2)
  plt.imshow(InputImage[:,:,1],cmap='Greys')
  plt.title('channel 2: green')
  plt.colorbar()
  
  plt.figure(3)
  plt.imshow(InputImage[:,:,2],cmap='Greys')
  plt.title('channel 3: blue')
  plt.colorbar()
  
  plt.show()


def showImage1channel(InputImage):
  imgplot = plt.imshow(InputImage,cmap='Greys')
  #imgplot = plt.imshow(InputImage)
  plt.colorbar()
  plt.show()


#-----------------------------------------------------------------------
#2) segmentation automatique d'image couleur avec l'algorithme des K-means
#-----------------------------------------------------------------------


Im2=matplotlib.pyplot.imread('TP1_Exo2_voiture.jpg')
Im=Im2*1.
showImage3channels(Im)


#QUESTION 1 : retransformer la forme de l'image pour qu'elle soit un array 
#2D de taille (M,3), ou M est le nombre de pixel dans l'image et 3 
#correspond au cannaux RGB dans l'image. Chaque pixel de l'image peut
#alors etre considere comme une observation en dimension 3.

#aide:
# -> pour connaitre la taille de l'image, on peut utiliser Im.shape
# -> pour changer la forme d'un numpy array, utiliser reshape



#QUESTION2 : Utiliser l'algorithme de K-means pour attribuer un label (segmenter)
#a chaque pixel de l'image

import scipy.cluster.vq as scipyvq

#aide:
# -> utiliser la fonction scipyvq.kmeans2



#QUESTION 3 : transformer la forme des labels pour quelle corresponde a la forme 
#de l'image initiale


#aide:
# -> utiliser reshape a nouveau


