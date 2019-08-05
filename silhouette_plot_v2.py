"""

Silhouette Plot adapted from:
https://gist.githubusercontent.com/clintval/e9afc246e77f6488cda79f86e4d37148/raw/23ea7565e74fd0b3c38afc50d25e1e6c609d68fd/kmeansplots.py 

With a fix from:
https://stackoverflow.com/questions/51452112/how-to-fix-cm-spectral-module-matplotlib-cm-has-no-attribute-spectral

Usage:
from silhouette_plot_v2 import silhouette_plot

"""

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter

__license__ = 'MIT'
__author__ = 'clintval'

def silhouette_plot(X, y, n_clusters, ax=None):
    """
    sihouette_plot(X,y,nclusters,ax=None)
    plot silhouette plots for clustering solutions
    
    X - numpy array of data with n rows of observations and p columns of variables clustered on
    y - predicted cluster labels, a [n,] one column array
    nclusters - integer number of clusters
    ax - matplotlib axis def,  Will use current specs by default
    
    """
    
    from sklearn.metrics import silhouette_samples, silhouette_score

    if ax is None:
        ax = plt.gca()

    # Compute the silhouette scores for each sample
    silhouette_avg = silhouette_score(X, y)
    sample_silhouette_values = silhouette_samples(X, y)

    y_lower = padding = 2
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        ith_cluster_silhouette_values = sample_silhouette_values[y == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

         # color = cm.spectral(float(i) / n_clusters)
        cmap = cm.get_cmap("Spectral")                #  Here's the fix referred to up top
        color = cmap(float(i) / n_clusters)           #  Here's more fix 
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0,
                         ith_cluster_silhouette_values,
                         facecolor=color,
                         edgecolor=color,
                         alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i + 1))

        # Compute the new y_lower for next plot
        y_lower = y_upper + padding

    ax.set_xlabel("The silhouette coefficient values")
    ax.set_ylabel("Cluster label")

    # The vertical line for average silhoutte score of all the values
    ax.axvline(x=silhouette_avg, color='r', alpha=0.8, lw=0.8, ls='-')
    ax.annotate('Average',
                xytext=(silhouette_avg, y_lower * 1.025),
                xy=(0, 0),
                ha='center',
                alpha=0.8,
                color='r')
               

    ax.set_yticks([])  # Clear the yaxis labels / ticks
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_ylim(0, y_upper + 1)
    ax.set_xlim(-0.075, 1.0)
    return ax
