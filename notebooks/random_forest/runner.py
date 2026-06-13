import sys
sys.path.append('/home/chotu/Projects/BareMetalML')

from sklearn.datasets import load_iris
from utils import train_test_split, StandardScaler
from random_forest import RandomForest

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

rf = RandomForest(n_trees=10, max_depth=5)
rf.fit(X_train, y_train)

print("Number of trees:", len(rf.trees))          
print("Max features used:", rf.max_features)      
print("First tree type:", type(rf.trees[0][0]))   
print("Feature indices sample:", rf.trees[0][1])  



preds = rf.predict(X_test)

from utils import accuracy_score
print("Random Forest Accuracy:", accuracy_score(y_test, preds))  # expect > 0.90

# compare against single tree
from notebooks.decision_tree.decision_tree import DecisionTree
single_tree = DecisionTree(max_depth=5)
single_tree.fit(X_train, y_train)
single_preds = single_tree.predict(X_test)
print("Single Tree Accuracy   :", accuracy_score(y_test, single_preds))