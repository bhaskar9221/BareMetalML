import numpy as np


def gini_impurity(y):
    
    counts = np.bincount(y)
    probabilities = counts / len(y)
    gini = 1 - np.sum(probabilities ** 2)
    return gini


def entropy(y):
    
    counts = np.bincount(y)
    probabilities = counts / len(y)
    probabilities = probabilities[probabilities > 0]
    entropy = -np.sum(probabilities * np.log2(probabilities))  
    return entropy


def information_gain(y, y_left, y_right, criterion='gini'):
    if criterion == 'gini':
        impurity_fn = gini_impurity
    else:
        impurity_fn = entropy
    
    parent_impurity   = impurity_fn(y)
    weighted_impurity = (len(y_left) * impurity_fn(y_left) + 
                         len(y_right) * impurity_fn(y_right)) / len(y)
    return parent_impurity - weighted_impurity


def find_best_split(X, y, criterion='gini'):
    best_gain      = -1
    best_feature   = None
    best_threshold = None

    n_features = X.shape[1]

    for feature_idx in range(n_features):
        thresholds = np.unique(X[:, feature_idx])

        for threshold in thresholds:
            left_mask  = X[:, feature_idx] < threshold
            right_mask = ~left_mask

            y_left  = y[left_mask]
            y_right = y[right_mask]

            if len(y_left) == 0 or len(y_right) == 0:
                continue

            gain = information_gain(y, y_left, y_right, criterion)

            if gain > best_gain:
                best_gain      = gain
                best_feature   = feature_idx
                best_threshold = threshold

    return best_feature, best_threshold


class Node:
    def __init__(self, feature=None, threshold=None, 
                 left=None, right=None, value=None):
        self.feature   = feature
        self.threshold = threshold
        self.left      = left
        self.right     = right
        self.value     = value

    def is_leaf(self):
        return self.value is not None


class DecisionTree:

    def __init__(self, max_depth=10, min_samples_split=2, criterion='gini'):
        
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None
        pass

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        n_samples  = len(y)
        n_classes  = len(np.unique(y))

        
        if depth >= self.max_depth or n_classes == 1 or n_samples < self.min_samples_split:
            leaf_value = np.bincount(y).argmax()
            return Node(value=leaf_value)

        feature, threshold = find_best_split(X, y, self.criterion)

        left_mask  = X[:, feature] < threshold
        right_mask = ~left_mask

        left  = self._build_tree(X[left_mask],  y[left_mask],  depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(feature=feature, threshold=threshold, left=left, right=right)

    def _predict_one(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] < node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])

    def print_tree(self, node=None, depth=0):
        if node is None:
            node = self.root

        indent = "  " * depth      # 2 spaces per depth level

        if node.is_leaf():
            print(f"{indent}Leaf → class {node.value}")
            return

        print(f"{indent}Feature {node.feature} <= {node.threshold:.2f}")
        
        print(f"{indent}  [Left]")
        self.print_tree(node.left,  depth + 1)
        print(f"{indent}  [Right]")
        self.print_tree(node.right, depth + 1)    

