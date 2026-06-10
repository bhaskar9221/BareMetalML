import numpy as np

def gini_impurity(y):
    
    unique_classes, counts = np.bincount(y, return_counts=True)
    probabilities = counts / len(y)
    gini = 1 - np.sum(probabilities ** 2)
    return gini

def entropy(y):
    
    counts = np.bincount(y)
    probabilities = counts / len(y)
    probabilities = probabilities[probabilities > 0]
    entropy = -np.sum(probabilities * np.log2(probabilities))  
    return entropy