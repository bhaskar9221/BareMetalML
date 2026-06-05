"""
Splitting the Data into Train and Test Sets for Machine Learning Models
"""

import numpy as np 
from sklearn.datasets import load_iris

def train_test_split(X, y, test_size=0.2, random_state=None):
    """Split data into train and test sets
       Splitting the Iris Dataset(for this demonstration) into training and test sets.)
    
    """

    #
    if random_state is not None:              
        np.random.seed(random_state)

    #Getting the row size from the Dataset
    n = X.shape[0]

    #Generating shuffeled indices and determining the train test split point.
    indices = np.arange(n)
    np.random.shuffle(indices)
    split = int(n * (1 - test_size))

    #Initializing the Features and Target Variables for the Train and Test Sets
    train_idx = indices[:split]
    test_idx = indices[split:]

    #Returning the final train and test sets
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]