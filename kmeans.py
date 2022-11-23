import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

data = pd.read_csv(sys.argv[1])
pca = PCA()
#pregunta al dr

datos = pca.fit_transform(data)
k = int(sys.argv[2])
kmeans = KMeans(n_clusters=k)
label = kmeans.fit_predict(datos)
centroids = kmeans.cluster_centers_
u_labels = np.unique(label)
for i in u_labels:
    plt.scatter(datos[label == i, 0], datos[label == i, 1],
    label=str("Cluster ") + str(i), cmap='tab10')


title = "K-Means de K = "+ str(k)
fig = plt.scatter(centroids[:, 0], centroids[:, 1], s=80, color='black', label='Centroides')
man = plt.get_current_fig_manager()
man.set_window_title(title)

plt.legend()
plt.show()
