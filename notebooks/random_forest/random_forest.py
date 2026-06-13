import sys
sys.path.append('/home/chotu/Projects/BareMetalML')
import numpy as np
from notebooks.decision_tree.decision_tree import DecisionTree


def bootstrap_sample(X, y):
    n = X.shape[0]
    indices = np.random.choice(n, size=n, replace=True)
    return X[indices], y[indices]


class RandomForest:

    def __init__(self, n_trees=10, max_depth=10, 
                 min_samples_split=2, max_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n_features = X.shape[1]

        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))

        for _ in range(self.n_trees):
            X_sample, y_sample = bootstrap_sample(X, y)

            feature_indices = np.random.choice(n_features, self.max_features, replace=False)

            X_sample_sub = X_sample[:, feature_indices]

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X_sample_sub, y_sample)
            self.trees.append((tree, feature_indices))
