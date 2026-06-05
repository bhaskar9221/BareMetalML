"""
This module is for Splitting the Data into Train and Test Sets for Machine Learning Models
"""

import numpy as np 
from sklearn.datasets import load_iris, fetch_california_housing, load_diabetes

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






###############Testing###################
if __name__ == "__main__":
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, y_train, X_test, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("X_train:", X_train.shape)
    print("X_test: ", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test: ", y_test.shape)

    print("\nClass distribution in train:", np.bincount(y_train))
    print("Class distribution in test: ", np.bincount(y_test))

    X_tr1, _, X_te1, _ = train_test_split(X, y, random_state=42)
    X_tr2, _, X_te2, _ = train_test_split(X, y, random_state=42)

    print("\nSame seed = same split:", np.array_equal(X_tr1, X_tr2))