"""
This modules contains functions for preprocessing the data, such as Scaling, MinMax Scaling !
"""

import numpy as np
from sklearn.datasets import load_iris, fetch_california_housing, load_diabetes



#Standard Scaler

class StandardScaler:
    
    def __init__(self):
        self.mean_ = None
        self.std_  = None
    
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)        # shape (n_features,)
        self.std_  = np.std(X, axis=0, ddof=0) # population std
        return self                             # allows chaining: scaler.fit(X).transform(X)
    
    def transform(self, X):

        std_safe = np.where(self.std_ == 0, 1, self.std_)  # avoid divide by zero
        return (X - self.mean_) / std_safe
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)


class MinMaxScaler:

    
    def __init__(self):
        self.min_ = None
        self.max_ = None
    
    def fit(self, X):
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        return self
    
    def transform(self, X):
        range_safe = np.where(
            (self.max_ - self.min_) == 0, 1, self.max_ - self.min_
        )
        return (X - self.min_) / range_safe
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)


###############Testing###################
if __name__ == "__main__":
    from data_utils import train_test_split
    
    # --- StandardScaler on California Housing ---
    # Housing has wildly different feature scales — perfect test
    print("=" * 45)
    print("StandardScaler — California Housing")
    print("=" * 45)
    housing = load_diabetes()
    X = housing.data
    print(f"\nBefore scaling:")
    print(f"  Mean per feature : {X.mean(axis=0).round(2)}")
    print(f"  Std  per feature : {X.std(axis=0).round(2)}")
    # you'll see wildly different numbers e.g. [3.87, 28.6, 5.4, 1.09, 1425, 3.07, 35.6, -119]

    X_train, X_test, _, _ = train_test_split(X, housing.target, random_state=42)

    scaler = StandardScaler()
    scaler.fit(X_train)                         # fit ONLY on train
    X_train_sc = scaler.transform(X_train)
    X_test_sc  = scaler.transform(X_test)       # uses train's mean/std

    print(f"\nAfter scaling (train):")
    print(f"  Mean per feature : {X_train_sc.mean(axis=0).round(4)}")  # all ~0.0
    print(f"  Std  per feature : {X_train_sc.std(axis=0).round(4)}")   # all ~1.0

    print(f"\nAfter scaling (test) — will NOT be exactly 0/1, that's correct:")
    print(f"  Mean per feature : {X_test_sc.mean(axis=0).round(4)}")
    print(f"  Std  per feature : {X_test_sc.std(axis=0).round(4)}")

    # --- MinMaxScaler on Iris ---
    print("\n" + "=" * 45)
    print("MinMaxScaler — Iris")
    print("=" * 45)
    iris = load_iris()
    X_train_i, X_test_i, _, _ = train_test_split(
        iris.data, iris.target, random_state=42
    )

    mm = MinMaxScaler()
    X_train_mm = mm.fit_transform(X_train_i)
    X_test_mm  = mm.transform(X_test_i)

    print(f"\nTrain min per feature: {X_train_mm.min(axis=0).round(4)}")  # all 0.0
    print(f"Train max per feature: {X_train_mm.max(axis=0).round(4)}")   # all 1.0
    print(f"Test  min per feature: {X_test_mm.min(axis=0).round(4)}")    # may go slightly below 0 — why?
    print(f"Test  max per feature: {X_test_mm.max(axis=0).round(4)}")    # may go slightly above 1 — why?