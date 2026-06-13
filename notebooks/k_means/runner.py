import sys
sys.path.append('/home/chotu/Projects/BareMetalML')

from notebooks.k_means.k_means import (
    KMeans,
    init_kmeans_plusplus,
    init_random
)

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
X = np.vstack([
    np.random.randn(50, 2) + [0, 0],
    np.random.randn(50, 2) + [8, 8],
    np.random.randn(50, 2) + [0, 8],
])

centroids_random = init_random(X, k=3)
centroids_pp = init_kmeans_plusplus(X, k=3)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].scatter(X[:, 0], X[:, 1], alpha=0.3)
axes[0].scatter(
    centroids_random[:, 0],
    centroids_random[:, 1],
    c='red',
    s=200,
    marker='X'
)
axes[0].set_title("Random Init")

axes[1].scatter(X[:, 0], X[:, 1], alpha=0.3)
axes[1].scatter(
    centroids_pp[:, 0],
    centroids_pp[:, 1],
    c='red',
    s=200,
    marker='X'
)
axes[1].set_title("K-Means++ Init")

plt.savefig("kmeans_init.png")

km = KMeans(k=3, max_iters=100)
km.fit(X)

print("Final centroids:\n", km.centroids)
print("Inertia:", km.inertia_)
print("Unique labels:", np.unique(km.labels_))

plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], c=km.labels_, cmap='viridis', alpha=0.5)
plt.scatter(
    km.centroids[:, 0],
    km.centroids[:, 1],
    c='red',
    s=200,
    marker='X'
)
plt.title("K-Means Final Clusters")
plt.savefig("kmeans_result.png")

inertias = []
ks = range(1, 11)

for k in ks:
    km = KMeans(k=k, max_iters=100)
    km.fit(X)
    inertias.append(km.inertia_)

plt.figure(figsize=(7, 4))
plt.plot(ks, inertias, marker='o')
plt.xlabel("k")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.savefig("kmeans_elbow.png")




X_path = np.array([
    [0, 0], [0, 0.1], [0, -0.1],
    [10, 10], [10, 10.1], [10, 9.9]
])

km_path = KMeans(k=5, max_iters=50)
km_path.fit(X_path)
print("Centroids:\n", km_path.centroids)
print("Labels:", km_path.labels_)
print("Unique labels used:", np.unique(km_path.labels_))
print("Any NaN?", np.isnan(km_path.centroids).any())

X_same = np.ones((20, 2))

km_same = KMeans(k=3, max_iters=50)
km_same.fit(X_same)
print("\nCentroids (all same data):\n", km_same.centroids)
print("Any NaN?", np.isnan(km_same.centroids).any())

