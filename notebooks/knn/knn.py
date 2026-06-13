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











import numpy as np

X = np.array([[1, 2], [4, 6], [0, 0]])
x1 = np.array([0, 0])

print(euclidean_distance(x1, X))   # [2.236, 7.211, 0.0]
print(manhattan_distance(x1, X))   # [3, 10, 0]

# timing test
X_big = np.random.randn(1000, 50)
x1_big = np.random.randn(50)

import time
start = time.time()
euclidean_distance(x1_big, X_big)
print("Euclidean time:", time.time() - start)

start = time.time()
manhattan_distance(x1_big, X_big)
print("Manhattan time:", time.time() - start)



from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

knn = KNNClassifier(k=5)
knn.fit(X_train, y_train)
preds = knn.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))   # expect > 0.90

# sweep k
import matplotlib.pyplot as plt
accs = []
for k in range(1, 21):
    knn = KNNClassifier(k=k)
    knn.fit(X_train, y_train)
    accs.append(accuracy_score(y_test, knn.predict(X_test)))