'''Cluster stocks based on historical prices'''
# pylint: disable=C0103
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def load_cluster(stock_data):
    '''Load and fit cluster from the data'''
    X = np.array(stock_data)

    cluster = KMeans(n_clusters=8)
    cluster.fit(X)

    labels = cluster.predict(X)
    print(labels)
    plt.scatter(x=X[:, -1], y=X[:, 1], c=labels)
    plt.show()

    return cluster, X
