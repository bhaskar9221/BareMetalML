import sys
sys.path.append('/home/chotu/Projects/BareMetalML')
import numpy as np

def init_random(X, k):
    indices = np.random.choice(X.shape[0], k, replace=False)
    return X[indices]

def init_kmeans_plusplus(X, k):
    n_samples = X.shape[0]

    first_idx = np.random.choice(n_samples)
    centroids = [X[first_idx]]

    for _ in range(k - 1):
        distances = np.array([
            min(np.sum((x - c) ** 2) for c in centroids)
            for x in X
        ])

        total = distances.sum()
        if total == 0:
            # all remaining points are identical to existing centroids
            # fall back to uniform random choice
            probabilities = np.ones(n_samples) / n_samples
        else:
            probabilities = distances / total

        next_idx = np.random.choice(n_samples, p=probabilities)
        centroids.append(X[next_idx])

    return np.array(centroids)


class KMeans:

    def __init__(self, k=3, max_iters=100, init='kmeans++'):
        self.k = k
        self.max_iters = max_iters
        self.init = init
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None

    def fit(self, X):
        if self.init == 'random':
            self.centroids = init_random(X, self.k)
        else:
            self.centroids = init_kmeans_plusplus(X, self.k)

        for _ in range(self.max_iters):
            distances = np.array([
                np.linalg.norm(X - c, axis=1)
                for c in self.centroids
            ])

            labels = np.argmin(distances, axis=0)

            new_centroids = np.array([
                X[labels == i].mean(axis=0)
                if np.any(labels == i)
                else X[np.random.choice(len(X))]
                for i in range(self.k)
            ])

            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

        self.labels_ = labels

        self.inertia_ = np.sum([
            np.sum((X[labels == i] - self.centroids[i]) ** 2)
            for i in range(self.k)
        ])

    def predict(self, X):
        distances = np.array([
            np.linalg.norm(X - c, axis=1)
            for c in self.centroids
        ])

        return np.argmin(distances, axis=0)