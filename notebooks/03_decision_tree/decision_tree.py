import numpy as np

def gini_impurity(y):
    
    unique_classes, counts = np.bincount(y, return_counts=True)
    probabilities = counts / len(y)
    gini = 1 - np.sum(probabilities ** 2)
    return gini