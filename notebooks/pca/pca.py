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

        self.all_eigenvalues_ = eigenvalues


        return self
    
    def transform(self, X):
        X_centered = X - self.mean_
        return X_centered @ self.components_.T
    
    
    def inverse_transform(self, X_reduced):
        return (X_reduced @ self.components_) + self.mean_


    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    @property
    def explained_variance_ratio_(self):
        return self.explained_variance_ / np.sum(self.all_eigenvalues_)