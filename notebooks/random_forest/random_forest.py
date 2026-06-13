import sys
sys.path.append('/home/chotu/Projects/BareMetalML')
import numpy as np
from notebooks.decision_tree.decision_tree import DecisionTree


def bootstrap_sample(X, y):
    n = X.shape[0]
    indices = np.random.choice(n, size=n, replace=True)
    return X[indices], y[indices]