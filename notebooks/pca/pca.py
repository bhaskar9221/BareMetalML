import sys
sys.path.append('/home/chotu/Projects/BareMetalML')
import numpy as np


class PCA:

    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.mean_ = None

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        n_samples = X.shape[0]
        cov = (X_centered.T @ X_centered) / (n_samples - 1)

        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]

        self.components_ = eigenvectors[:, :self.n_components].T
        self.explained_variance_ = eigenvalues[:self.n_components]

        return self