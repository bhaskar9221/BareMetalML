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




