'''Cluster stocks based on historical prices'''
# pylint: disable=C0103
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt


sns.set(style="white", palette="muted", color_codes=True)

def load_cluster(stock_data, n_clusters=5):
    '''Load and fit cluster from the data'''
    X = preprocessing.scale(stock_data)

    # Calculate the silhouette_score with cluster size and select most appropriate

    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    lables = kmeans.labels_

    silhouette_avg = silhouette_score(X, lables)
    #print("For n_clusters ={}\nThe average silhouette_score is :{}".format(n_clusters,
    #                                                                      silhouette_avg))

    # Fit and Transform PCA on the given data set to reduce number of columns to 2 for plotting
    pca_stock = PCA(n_components=2).fit_transform(X)
    pca_stock = pd.DataFrame(pca_stock, index=stock_data.index.copy(), columns=['x_axis', 'y_axis'])

    pca_stock['Lables'] = lables

    _, ax = plt.subplots()
    #Scatterplot with stocks as data points
    ax.scatter(x=pca_stock.ix[:, 0], y=pca_stock.ix[:, 1], c=lables)
    #Adding annotation to each point to show the stock name and cluster name
    for i, txt in enumerate(pca_stock.index):
        ax.annotate(txt, (pca_stock.ix[i, 0], pca_stock.ix[i, 1]), size=10)

    plt.show()
