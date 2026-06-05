"""
This module contains functions for plotting training curves and decision boundaries.
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, fetch_california_housing, load_diabetes

def plot_decision_boundary(model, X, y, title="Decision Boundary", resolution=0.02):
   
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution)
    )

    grid        = np.c_[xx1.ravel(), xx2.ravel()]
    Z           = model.predict(grid)
    Z           = Z.reshape(xx1.shape)

    plt.figure(figsize=(8, 5))
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap='RdYlBu')
    scatter = plt.scatter(
        X[:, 0], X[:, 1],
        c=y, cmap='RdYlBu',
        edgecolors='black', s=40, linewidths=0.5
    )
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.tight_layout()
    plt.show()


###############Testing###################
if __name__ == "__main__":

    class DummyModel:
        def fit(self, X, y):
            self.threshold = np.median(X[:, 0])
        def predict(self, X):
            return (X[:, 0] > self.threshold).astype(int)

    iris  = load_iris()
    X     = iris.data[:, :2]            
    y     = iris.target

    model = DummyModel()
    model.fit(X, y)

    plot_decision_boundary(
        model, X, y,
        title="Dummy Model — splits on Feature 0 median"
    )