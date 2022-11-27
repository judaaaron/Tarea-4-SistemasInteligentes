import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn.cluster import DBSCAN

nombres = ["datos_1.csv", "datos_2.csv", "datos_3.csv"]
min_sampless = [5, 10, 15]
epsVals = [0.01, 0.1, 0.3, 0.5, 1]

for nombre in nombres:
    for min_samples in min_sampless:
        for epsVal in epsVals:
            datasetNumber = nombre.split("_")[1].split(".")[0]
            datos = pd.read_csv(nombre)
            datos = np.array(datos)

            db = DBSCAN(eps=float(epsVal), min_samples=int(min_samples)).fit(datos)
            labels = db.labels_
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)
            unique_labels = set(labels)
            colors = [plt.cm.Spectral(each)
                    for each in np.linspace(0, 1, len(unique_labels))]

            y_pred = db.fit_predict(datos)

            title = "DBSCAN de eps = " + \
                str(epsVal) + " y min_samples = " + str(min_samples)
            fig = plt.scatter(datos[:, 0], datos[:, 1], c=y_pred, cmap='rainbow')
            man = plt.get_current_fig_manager()
            man.set_window_title(title)

            #TODO: cambiar de mostrar el plot a guardar el plot en un archivo
            plt.savefig("./Figuras/DBSCAN/DBSCAN_eps_" + str(epsVal) + "_min_samples_" + str(min_samples) + "_data_" + datasetNumber + ".png")
            plt.close()
