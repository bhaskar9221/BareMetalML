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






################Testing###################
if __name__ == "__main__":
    
    # --- Regression: California Housing ---
    print("=" * 40)
    print("REGRESSION METRICS — California Housing")
    print("=" * 40)
    housing = fetch_california_housing()
    y = housing.target                          # median house values

    # simulate a "dumb" model that always predicts the mean
    y_pred_mean  = np.full_like(y, y.mean())
    # simulate a "slightly smart" model with small noise
    y_pred_smart = y + np.random.normal(0, 0.5, size=y.shape)

    print(f"\nDumb model (always predicts mean):")
    print(f"  MSE : {mean_squared_error(y, y_pred_mean):.4f}")
    print(f"  MAE : {mean_absolute_error(y, y_pred_mean):.4f}")
    print(f"  R²  : {r2_score(y, y_pred_mean):.4f}")   # should be ~0.0

    print(f"\nSmart model (y + small noise):")
    print(f"  MSE : {mean_squared_error(y, y_pred_smart):.4f}")
    print(f"  MAE : {mean_absolute_error(y, y_pred_smart):.4f}")
    print(f"  R²  : {r2_score(y, y_pred_smart):.4f}")  # should be close to 1.0

    # --- Classification: Iris (binary: class 0 vs class 1 only) ---
    print("\n" + "=" * 40)
    print("CLASSIFICATION METRICS — Iris (binary)")
    print("=" * 40)
    iris = load_iris()
    # keep only class 0 and 1 for binary metrics
    mask   = iris.target < 2
    y_true = iris.target[mask]                  # 100 samples, labels 0 and 1
    # simulate predictions with a few errors
    y_pred = y_true.copy()
    y_pred[5]  = 1 - y_pred[5]                 # flip a few
    y_pred[20] = 1 - y_pred[20]
    y_pred[75] = 1 - y_pred[75]

    print(f"\nAccuracy : {accuracy_score(y_true, y_pred):.4f}")   # ~0.97
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred):.4f}")
    print(f"F1       : {f1_score(y_true, y_pred):.4f}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_true, y_pred)}")
    print("Rows = Actual, Columns = Predicted")

    # --- Multiclass confusion matrix ---
    print("\n" + "=" * 40)
    print("CONFUSION MATRIX — Iris (3-class)")
    print("=" * 40)
    y_true3 = iris.target
    y_pred3 = y_true3.copy()
    y_pred3[10] = 2                             # wrong prediction
    y_pred3[60] = 0
    y_pred3[110] = 1
    print(confusion_matrix(y_true3, y_pred3))
    # diagonal should be near-perfect, 3 off-diagonal errors