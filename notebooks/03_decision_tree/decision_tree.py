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