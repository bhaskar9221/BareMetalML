"""
Splitting the Data into Train and Test Sets for Machine Learning Models
"""

import numpy as np 
from sklearn.datasets import load_iris

def train_test_split(X, y, test_size=0.2, random_state=None):
    """Split data into train and test sets
       Splitting the Iris Dataset(for this demonstration) into training and test sets.)
    
    """
    if random_state is not None:
        np.random.seed(random_state)

    n = X.shape[0]

    indices = np.arange(n)
    np.random.shuffle(indices)
    split = int(n * (1 - test_size))

    train_idx = indices[:split]
    test_idx = indices[split:]

    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]

