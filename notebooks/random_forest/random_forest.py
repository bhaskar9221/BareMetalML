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


    def predict(self, X):
        
        all_preds = []

        for tree, feature_indices in self.trees:
            X_sub = X[:, feature_indices]
            preds = tree.predict(X_sub)
            all_preds.append(preds)

        all_preds = np.array(all_preds)   

        final_preds = []
        for i in range(X.shape[0]):
            votes = all_preds[:, i]
            majority = np.bincount(votes).argmax()   
            final_preds.append(majority)

        return np.array(final_preds)
    
    @property
    def feature_importances_(self):
        n_features = max(max(fi) for _, fi in self.trees) + 1
        importances = np.zeros(n_features)

        for tree, feature_indices in self.trees:
            for local_idx, global_idx in enumerate(feature_indices):
                importances[global_idx] += tree.feature_importances_[local_idx]

        importances /= self.n_trees
        return importances / np.sum(importances)