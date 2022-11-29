import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn.cluster import AgglomerativeClustering



nombres = ["./Datasets/datos_1.csv", "./Datasets/datos_2.csv", "./Datasets/datos_3.csv"]
ks = [1,2,3,4,5]
tipos = ['u', 'c']

for nombre in nombres:
    for k in ks:
        for tipo in tipos:
            data = pd.read_csv(nombre)
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
                title = "./Figuras/Agglomerative/Agglomerative_k_None-umbral_" + str(k)+ "_data_" + nombre.split("_")[1].split(".")[0] + ".png"
            elif(tipo == 'c'):
                title = "./Figuras/Agglomerative/Agglomerative_k_clusters" + str(k) + "_umbral_None"+ "_data_" + nombre.split("_")[1].split(".")[0] + ".png"
            fig = plt.scatter(datos[:, 0], datos[:, 1], c=labels, cmap='rainbow')
            plt.legend()
            plt.savefig(title)
