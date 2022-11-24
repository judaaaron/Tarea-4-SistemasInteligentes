import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn.cluster import AgglomerativeClustering
import argparse
from sklearn.decomposition import PCA

data = pd.read_csv(sys.argv[1])
k = sys.argv[2]

# parser = argparse.ArgumentParser()
# parser.add_argument(umbral)
# args = parser.parse_args()
#py ./jerarquico.py ./data1.csv 4 

tipo = sys.argv[3]
if(tipo == 'u'):
    clustering = AgglomerativeClustering(
        n_clusters=None, affinity='euclidean', linkage='ward', distance_threshold=int(k))
elif(tipo == 'c'):
    clustering = AgglomerativeClustering(
        n_clusters=int(k), affinity='euclidean', linkage='ward', distance_threshold=None)
else:
    print("No se ingresaron los parametros de manera correcta")

# try:
#     k = int(sys.argv[2])
# except IndexError:
#     k = None


    
    
data = np.array(data)
pca = PCA(2)
datos = pca.fit_transform(data)

# clustering = AgglomerativeClustering(
#     n_clusters=k, affinity='euclidean', linkage='ward', distance_threshold=None)

label = clustering.fit_predict(datos)
chiledren = clustering.children_
u_labels = np.unique(label)

for i in u_labels:
    plt.scatter(datos[label == i, 0], datos[label == i, 1],
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
