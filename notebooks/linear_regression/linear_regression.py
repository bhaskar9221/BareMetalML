import sys
sys.path.append('/home/chotu/Projects/BareMetalML')  
import numpy as np
from sklearn.datasets import load_diabetes
from utils import train_test_split, r2_score, StandardScaler
import matplotlib.pyplot as plt



class LinearRegression:

    def __init__(self, solver='gd', learning_rate = 0.01, n_iterations = 1000, alpha = 0.0):
        self.solver = solver
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.loss_history = []
        self.alpha = alpha

    def _add_bias(self, X):
        ones = np.ones((X.shape[0], 1))
        return np.hstack((ones, X))

    def fit(self, X, y):
        if self.solver == 'normal':
            self._fit_normal(X, y)
        elif self.solver == 'gd':
            self._fit_gd(X, y)

    def _fit_normal(self, X, y):
        X_b = self._add_bias(X)
        self.weights = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        
    
    def predict(self, X):
        X_b = self._add_bias(X)
        return X_b.dot(self.weights)

    def _fit_gd(self, X, y):
        X_b = self._add_bias(X)
        n, d = X_b.shape
        self.weights = np.zeros(d)
        self.loss_history = []

        for i in range(self.n_iterations):
            y_hat = X_b @ self.weights
            error = y_hat - y
            grad  = (2 / n) * X_b.T @ error
            grad = (2 / n) * X_b.T @ error
            grad = np.clip(grad, -1e10, 1e10)   

            if self.alpha > 0:
                reg    = 2 * self.alpha * self.weights.copy()
                reg[0] = 0.0
                grad  += reg

            self.weights -= self.learning_rate * grad
            self.loss_history.append(np.mean(error ** 2))
