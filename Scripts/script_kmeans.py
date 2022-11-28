import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np
from sklearn.cluster import KMeans

datos = ["./Datasets/datos_1.csv", "./Datasets/datos_2.csv", "./Datasets/datos_3.csv"]
ks = [1,2,3,4,5]

for nombre in datos:
    for k in ks:
        data = pd.read_csv(nombre)
        data = np.array(data)
        #pregunta al dr
        kmeans = KMeans(n_clusters=k)
        label = kmeans.fit_predict(data)
        print(label)
        centroids = kmeans.cluster_centers_
        u_labels = np.unique(label)
        for i in u_labels:
            Slabel = label == i
            plt.scatter(data[Slabel, 0], data[Slabel, 1],
            label=str("Cluster ") + str(i), cmap='tab10')


        title = "K-Means de K = "+ str(k)
        fig = plt.scatter(centroids[:, 0], centroids[:, 1], s=80, color='black', label='Centroides')
        man = plt.get_current_fig_manager()
        man.set_window_title(title)

        plt.legend()
        plt.savefig("./Figuras/K-Means/K-Means_" + nombre + "_K_" + str(k) + ".png")
        plt.close()
