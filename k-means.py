import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.cluster import KMeans

#prétraitement sur l'image
im = cv2.imread('image\elephant.jpg')

im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
original_shape = im.shape

plt.imshow(im)



# donne une matrice de n ligne(dépant de la taille de l'image) et 3 colonne pour les 3 pixels RGB array(num_pixels,3)
# all_pixels représente notre dataset
#
all_pixels = im.reshape((-1, 3))

###### k-means ##############

# choix de k
dominant_colors = 5

km = KMeans(n_clusters=dominant_colors)
#géneration du modéle
km.fit(all_pixels)

centers = km.cluster_centers_

# stocker les centres dans un tableau
centers = np.array(centers, dtype='uint8')



i = 1

# dimension de figure hauteur et largeur
plt.figure(0, figsize=(8, 2))

# list de color
colors = []

# boucler sur les coleurs
for each_col in centers:
    #création d'un sous-pacerelle
    plt.subplot(1,dominant_colors,i)
    plt.axis("off")
    i+=1

    #ajouter à la liste
    colors.append(each_col)

    # tab de 100 lign et 100 col pour rgb
    a = np.zeros((100, 100, 3), dtype='uint8')
    #attribue  each_col à tab
    a[:,:,:] = each_col
    plt.imshow(a)


new_img = np.zeros((im.shape[0]*im.shape[1], 3), dtype='uint8')


for ix in range(new_img.shape[0]):
    new_img[ix] = colors[km.labels_[ix]]

#image final
new_img = new_img.reshape((original_shape))
plt.figure(0, figsize=(8, 2))
#plt.imshow(new_img)
#cv2.imwrite('res.jpg',new_img)
plt.show()
plt.imshow(new_img)
plt.show()