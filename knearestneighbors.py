import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle as pkl
import time


data = pd.read_csv(sys.argv[1])
y = data.pop('class')
data = data.replace({"Si": 1, "No": 0})
data = np.array(data)

preName = sys.argv[1].split(".")[1]
preName = preName.split("_")[2]
clasificador = KNeighborsClassifier(n_neighbors=int(sys.argv[2]))
start = time.time()
clasificador.fit(data, y)
end = time.time()

y_pred = clasificador.predict(data)
val = accuracy_score(y_true=y, y_pred=y_pred)

title = "./ModelosKNN/KNN_k_" + sys.argv[2] + "_" + preName + ".pkl"
objeto = {"modelo": clasificador, "k": sys.argv[2], "trainingTime": end-start, "trainingAcc": val, "size": preName}
with open(title, 'wb') as file:
    pkl.dump(objeto, file)

print("Modelo guardado en:", title)