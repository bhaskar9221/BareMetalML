import sys
sys.path.append('/home/chotu/Projects/BareMetalML')
import numpy as np
from notebooks.decision_tree.decision_tree import DecisionTree


def bootstrap_sample(X, y):
    n = X.shape[0]
    indices = np.random.choice(n, size=n, replace=True)
    return X[indices], y[indices]


#####Testing#####

X = np.arange(100).reshape(50, 2)
y = np.arange(50)

X_boot, y_boot = bootstrap_sample(X, y)

print(X_boot.shape)               
print(y_boot.shape)               
print(len(np.unique(y_boot)))     