# !/usr/bin/python
# -*- coding: UTF-8 -*-

##########################
# Creator: Javy
# Create Time: 20170516
# Email: xujavy@gmail.com
# Description: Machine Learning - Chapter eleven
##########################

# Sample one
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=150, n_features=2,
                  centers=3, cluster_std=0.5,
                  shuffle=True, random_state=0)

import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1],
            c='white', marker='o', s=50)
plt.grid()
plt.show()

from sklearn.cluster import KMeans
km = KMeans(n_cluster=3, init='random', n_init=10,
            max_iter=300, tol=le-04, random_state=0)
y_km = km.fit_predict(X)

plt.scatter(X[y_km==0, 0], X[y_km=0, 1],
            s=50, c='lightgreen', marker='s', label='cluster 1')
plt.scatter(X[y_km==1, 0], X[y_km=1, 1],
            s=50, c='orage', marker='o', label='cluster 1=2')
plt.scatter(X[y_km==2, 0], X[y_km=2, 1],
            s=50, c='lightblue', marker='v', label='cluster 3')
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
            s=250, c='red', marker='*', label='centroids')
plt.grid()
plt.legend()
plt.show()

print('Distortion: {0}'.format(km.inertia_))

distortions = []
for i range(1, 11):
    km = KMeans(n_cluster=i, init='k-means++',
                n_init=10, max_iter=300, random_state=0)
    km.fit(X)
    distortions.append(km.inertia_)
plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Number of cluster')
plt.ylabel('Distortion')
plt.show()

# Sample two
km = KMeans(n_cluster=3, init='random', n_init=10,
            max_iter=300, tol=le-04, random_state=0)
y_km = km.fit_predict(X)

import numpy as np
from matplotlib import cm
from sklearn.metrics import silhouette_samples
cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')
y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
    c_silhouette_vlus = silhouette_vals[y_km == c]
    c_silhouette_vlus.sort()
    y_ax_upper += len(c_silhouette_vlus)
    color = cm.jet(i / n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vlus,
             height=1.0, edgecolor='none', color=color)
    yticks.append((y_ax_lower + y_ax_upper) / 2)
    y_ax_lower += len(c_silhouette_vlus)
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, color='red',
            linestyle='--')
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')
plt.show()

# Sample three
km = KMeans(n_cluster=2, init='random', n_init=10,
            max_iter=300, tol=le-04, random_state=0)
y_km = km.fit_predict(X)

plt.scatter(X[y_km==0, 0], X[y_km==0, 1],
            s=50, c='lightgreeen', marker='s', label='cluster 1')
plt.scatter(X[y_km==1, 0], X[y_km==1, 1],
            s=50, c='orage', marker='o', label='cluster 2')
plt.scatter(km,cluster_centers_[:, 0], km.cluster_centers_[:, 1],
            s=250, c='red', marker='*', label='centroids')
plt.legned()
plt.grid()
plt.show()


cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')
y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
    c_silhouette_vlus = silhouette_vals[y_km == c]
    c_silhouette_vlus.sort()
    y_ax_upper += len(c_silhouette_vlus)
    color = cm.jet(i / n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vlus,
             height=1.0, edgecolor='none', color=color)
    yticks.append((y_ax_lower + y_ax_upper) / 2)
    y_ax_lower += len(c_silhouette_vlus)
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, color='red',
            linestyle='--')
plt.yticks(yticks, cluster_labels + 1)
plt..yticks('yticks, cluster_labels + 1')
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')
plt.show()

# Sample four
import pandas as pd
import numpy as np
np.random.seed(123)
variables = ['X', 'Y', 'Z']
labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4']
X = np.random.random_sample([5, 3])*10
df = pd.DataFrame(X, columns=variables, index=labels)
df

from scipy.saptial.distance import pdist, squareform
row_dist = pd.DataFrame(squareform(pdist(df, metric='euclidean')),
                        columns=labels, index=labels)
row_dist

from scipy.cluster.hierarchy import linkage
help(linkage)
row_clusters = linkage(row_dist, method='complete',
                       metric='euclidean')
row_clusters = linkage(pdist(df, metric='euclidean'), method='complete')
row_clusters = linkage(df.values, method='complete', metric='euclidean')

pd.DataFrame(row_clusters,
             columns=['row label 1', 'row lable 2',
             'distance', 'no. of items in clust.'],
             index=['cluster {0}'.format(
                       (x+1) for i in range(row_clusters.shape[0]))])

from sklearn.cluster.hierarchy import dendrogram
row_dendr = dendrogram(row_clusters, label=labels,
                       # make dendrogram black (part 2/2)
                       # color_threshold=np.inf
                       )
plt.tight_layout()
plt.ylabel('Eucliden distance')
plt.show()

# Sample six
fig = plt.figure(figsize=(8, 8))
axd = fig.add_axes([0.09, 0.1, 0.2, 0.6])
row_dendr = dendrogram(row_clusters, oritentation='right')

df_rowclust = df.ix[row_dendr['leaves'][::-1]]
axm = fig.add_axes([0.23, 0.1, 0.6, 0.6])
cax = axm.matshow(df_rowclust, interpolation='nearest', cmp='hot_r')

axd.set_xticks([])
axd.set_yticks([])
for i in axd.spines.values():
    i.set_visible(False)
fig.colorbar(cax)
axm.set_xticklables([''] + list(df_rowclust.columns))
axm.set_yticklables([''] + list(df_rowclust.index))
plt.show()

# Sample seven
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_cluster=2, affinity='euclidean',
                             linkage='complete')
labels = ac.fit_predict(X)
print('Cluster labels: {0}'.format(labels))

from sklearn.datasets import make_moons
X, y = make_moons(n_samples=200, nosie=0.05,
                  random_state=0)
plt.scatter(X[:, 0], X[:, 1])
plt.show()

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,3))
km = KMeans(n_cluster=2, random_state=0)
y_km = km.fit_predict(X)
ax1.scatter(X[y_km==0, 0], X[y_km==0, 1],
            c='lightblue', marker='o', s=40, label='cluster 1')
ax1.scatter(X[y_km==1, 0], X[y_km==1, 1],
            c='red', marker='s', s=40, label='cluster 2')
ax1.set_title('K-means clustering')
ac = AgglomerativeClustering(n_cluster=2, affinity='euclidean', linkage='complete')
y_ac = ac.fit_predict(X)
ax1.scatter(X[y_ac==0, 0], X[y_ac==0, 1],
            c='lightblue', marker='o', s=40, label='cluster 1')
ax1.scatter(X[y_ac==1, 0], X[y_ac==1, 1],
            c='red', marker='s', s=40, label='cluster 2')
ax1.set_title('Agglomerative clustering')
plt.legend()
plt.show()

# Sample nine
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.2, min_samples=5, metric='euclidean')
y_db = db.fit_predict(X)
plt.scatter(X[y_db==0, 0], X[y_db==0, 1],
            c='lightblue', marker='o', s=40, label='cluster 1')
plt.scatter(X[y_db==1, 0], X[y_db==1, 1],
            c='red', marker='s', s=40, label='cluster 2')
plt.set_title('Agglomerative clustering')
plt.legend()
plt.show()
