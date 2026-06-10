
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

tree = DecisionTree(max_depth=3)   
tree.fit(X_train, y_train)
tree.print_tree()