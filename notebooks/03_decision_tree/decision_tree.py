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





#####Testing#####
pure = np.array([0, 0, 0, 0])
print(gini_impurity(pure))   # 0.0
print(entropy(pure))         # 0.0

mixed = np.array([0, 0, 1, 1])
print(gini_impurity(mixed))  # 0.5
print(entropy(mixed))        # 1.0

y        = np.array([0, 0, 1, 1])
y_left   = np.array([0, 0])
y_right  = np.array([1, 1])
print(information_gain(y, y_left, y_right))  # 0.5 — perfect split gains maximum