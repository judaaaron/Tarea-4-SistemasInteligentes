import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn.cluster import AgglomerativeClustering

data = pd.read_csv(sys.argv[1])
k = sys.argv[2]

tipo = sys.argv[3]
if(tipo == 'u'):
    clustering = AgglomerativeClustering(
        n_clusters=None, affinity='euclidean', linkage='ward', distance_threshold=int(k), compute_distances=True)
elif(tipo == 'c'):
    clustering = AgglomerativeClustering(
        n_clusters=int(k), affinity='euclidean', linkage='ward', distance_threshold=None, compute_distances=True)
else:
    print("No se ingresaron los parametros de manera correcta")

datos = np.array(data)

clusters = clustering.fit(datos)
labels = clusters.labels_

if(tipo == 'u'):
    title = "Agglomerative con k = None y umbral = " + k
elif(tipo == 'c'):
    title = "Agglomerative con k clusters = " + k + " y umbral = None"
fig = plt.scatter(datos[:, 0], datos[:, 1], c=labels, cmap='rainbow')
man = plt.get_current_fig_manager()
man.set_window_title(title)

plt.legend()
plt.show()
