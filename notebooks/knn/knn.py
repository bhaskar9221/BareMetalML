import numpy as np


def euclidean_distance(x1, X):
    return np.sqrt(np.sum((X - x1) ** 2, axis=1))

def manhattan_distance(x1, X):
    return np.sum(np.abs(X - x1), axis=1)













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