import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np
from sklearn.cluster import KMeans

data = pd.read_csv(sys.argv[1])
data = np.array(data)
#pregunta al dr
k = int(sys.argv[2])
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
plt.show()
