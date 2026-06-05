import numpy as np
from sklearn.datasets import fetch_california_housing, load_iris

"""
This modules contains functions for computing metrics related to Classification and Regression Problems
"""



"""
Regression Metrics
"""

#Mean Squared Error
def mean_squared_error(y_true, y_pred):

    return np.mean((y_true - y_pred) ** 2)


#Mean Absolute Error
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

#R2 Score
def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)




"""
Classification Metrics
"""

#Accuracy Score
def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

#Confusion Matrix
def confusion_matrix(y_true, y_pred):
    classes = np.unique(y_true)
    n = len(classes)
    matrix = np.zeros((n, n), dtype=int)
    for i, actual in enumerate(classes):
        for j, predicted in enumerate(classes):
            matrix[i, j] = np.sum((y_true == actual) & (y_pred == predicted))
    
    return matrix


#Precision Score(Binary Currently)
def precision_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


#Recall Score(Binary Currently)
def recall_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)


#F1 Score
def f1_score(y_true, y_pred):
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)

    if p + r == 0:
        return 0.0
    return 2 * (p * r) / (p + r)
