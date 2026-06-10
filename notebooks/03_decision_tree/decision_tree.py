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


#####Testing#####
pure = np.array([0, 0, 0, 0])
print(gini_impurity(pure))   
print(entropy(pure))         

mixed = np.array([0, 0, 1, 1])
print(gini_impurity(mixed))  
print(entropy(mixed))        

y        = np.array([0, 0, 1, 1])
y_left   = np.array([0, 0])
y_right  = np.array([1, 1])
print(information_gain(y, y_left, y_right))  

from sklearn.datasets import load_iris
iris   = load_iris()
X, y   = iris.data, iris.target

feature, threshold = find_best_split(X, y)
print(f"Best feature : {feature}")              
print(f"Best threshold: {threshold:.2f}")       
print(f"Feature name  : {iris.feature_names[feature]}")  


import sys
sys.path.append('/home/chotu/Projects/BareMetalML')
from sklearn.datasets import load_iris
from utils import train_test_split, accuracy_score, StandardScaler

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

tree = DecisionTree(max_depth=5)
tree.fit(X_train, y_train)
preds = tree.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))   
print("Root split — feature:", tree.root.feature)   
print("Root threshold:", tree.root.threshold)