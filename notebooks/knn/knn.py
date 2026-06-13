import numpy as np
import sys
sys.path.append('/home/chotu/Projects/BareMetalML')  
from utils import train_test_split, accuracy_score, StandardScaler


def euclidean_distance(x1, X):
    return np.sqrt(np.sum((X - x1) ** 2, axis=1))

def manhattan_distance(x1, X):
    return np.sum(np.abs(X - x1), axis=1)

class KNNClassifier:
    def __init__(self, k=5, metric='euclidean'):
        self.k = k
        self.metric = metric
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def _predict_one(self, x):
        if self.metric == 'euclidean':
            distances = euclidean_distance(x, self.X_train)
        elif self.metric == 'manhattan':
            distances = manhattan_distance(x, self.X_train)
        else:
            raise ValueError("metric must be 'euclidean' or 'manhattan'")
        k_indices = np.argsort(distances)[:self.k]
        k_labels = self.y_train[k_indices]
        return np.bincount(k_labels).argmax()
        
    def predict(self, X):
        return np.array([self._predict_one(x) for x in X])


class KNNRegressor:
    def __init__(self, k=5, metric='euclidean'):
        self.k = k
        self.metric = metric

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def _predict_one(self, x):
        if self.metric == 'euclidean':
            distances = euclidean_distance(x, self.X_train)
        elif self.metric == 'manhattan':
            distances = manhattan_distance(x, self.X_train)
        else:
            raise ValueError("metric must be 'euclidean' or 'manhattan'")
        k_indices = np.argsort(distances)[:self.k]
        k_values = self.y_train[k_indices]
        return np.mean(k_values)

    def predict(self, X):
        return np.array([self._predict_one(x) for x in X])