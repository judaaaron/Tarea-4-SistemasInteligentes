import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn.cluster import DBSCAN

datos = pd.read_csv(sys.argv[1])
epsVal = sys.argv[2]
min_samples = sys.argv[3]
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

# for k, col in zip(unique_labels, colors):
#     if k == -1:
#         # Black used for noise.
#         col = [0, 0, 0, 1]

#     class_member_mask = labels == k

#     xy = datos[class_member_mask & core_samples_mask]
#     plt.plot(
#         xy[:, 0],
#         xy[:, 1],
#         "o",
#         markerfacecolor=tuple(col),
#         markeredgecolor="k",
#         markersize=14,
#     )

#     xy = datos[class_member_mask & ~core_samples_mask]
#     plt.plot(
#         xy[:, 0],
#         xy[:, 1],
#         "o",
#         markerfacecolor=tuple(col),
#         markeredgecolor="k",
#         markersize=6,
#     )


y_pred = db.fit_predict(datos)

title = "DBSCAN de eps = " + \
    str(sys.argv[2]) + " y min_samples = " + str(sys.argv[3])
fig = plt.scatter(datos[:, 0], datos[:, 1], c=y_pred, cmap='rainbow')
man = plt.get_current_fig_manager()
man.set_window_title(title)

plt.show()
