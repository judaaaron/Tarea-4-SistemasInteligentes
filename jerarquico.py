import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
import argparse
from sklearn.decomposition import PCA

data = pd.read_csv(sys.argv[1])
k = sys.argv[2]


tipo = sys.argv[3]
if(tipo == 'u'):
    clustering = AgglomerativeClustering(
        n_clusters=None, affinity='euclidean', linkage='ward', distance_threshold=int(k))
elif(tipo == 'c'):
    clustering = AgglomerativeClustering(
        n_clusters=int(k), affinity='euclidean', linkage='ward', distance_threshold=None)
else:
    print("No se ingresaron los parametros de manera correcta")

datos = np.array(data)
n_clusters = datos.shape[0]

precomputed_data = np.zeros((n_clusters, n_clusters))
x=datos[:,0]
y = list()

for i in range(n_clusters):
    y.append(i)
y = np.array(y)

for i in range(n_clusters):
    for j in range(n_clusters):
        precomputed_data = pairwise_distances(x[y == i], x[y == j], metric='precomputed')

label = clustering.fit_predict(precomputed_data)
chiledren = clustering.children_
u_labels = np.unique(label)

for i in u_labels:
    plt.scatter(precomputed_data[label == i, 0], precomputed_data[label == i, 1],
                label=str("Cluster ") + str(i), cmap='rainbow')


if(tipo == 'u'):
    title = "Agglomerative con k = None y umbral = " + k
elif(tipo == 'c'):
    title = "Agglomerative con k clusters = " + k + " y umbral = None"
# fig = plt.scatter(chiledren[:, 0], chiledren[:, 1], s=80, color='black', label='chiledren')
man = plt.get_current_fig_manager()
man.set_window_title(title)

plt.legend()
plt.show()
