import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle as pkl
import time

ks = [1,3,5,7,9,11,13,15]
datasets = ["./training_data_small.csv", "./training_data_medium.csv", "./training_data_large.csv", "./training_data_very_large.csv"]

for nombre in datasets:
    for k in ks:
        data = pd.read_csv(nombre)
        y = data.pop('class')
        data = data.replace({"Si": 1, "No": 0})
        data = np.array(data)

        preName = nombre.split(".")[1]
        preName = preName.split("_")[2]
        clasificador = KNeighborsClassifier(n_neighbors=int(k))
        start = time.time()
        clasificador.fit(data, y)
        end = time.time()

        y_pred = clasificador.predict(data)
        val = accuracy_score(y_true=y, y_pred=y_pred)

        title = "./ModelosKNN/KNN_k_" + str(k) + "_" + preName + ".pkl"
        objeto = {"modelo": clasificador, "k": k, "trainingTime": end-start, "trainingAcc": val, "size": preName}
        with open(title, 'wb') as file:
            pkl.dump(objeto, file)

        print("Modelo guardado en:", title)