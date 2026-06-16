import sys
sys.path.append('/home/chotu/Projects/BareMetalML')
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.datasets import load_iris
from utils import train_test_split, accuracy_score, StandardScaler, plot_decision_boundary
from notebooks.logistic_regression.logistic_regression import OneVsRestClassifier
from notebooks.decision_tree.decision_tree import DecisionTree
from notebooks.random_forest.random_forest import RandomForest
from notebooks.knn.knn import KNNClassifier
from notebooks.naive_bayes.naive_bayes import GaussianNB


iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)


models = {
    "Logistic Regression": OneVsRestClassifier(learning_rate=0.1, n_iterations=1000),
    "Decision Tree": DecisionTree(max_depth=5),
    "Random Forest": RandomForest(n_trees=10, max_depth=5),
    "KNN (k=5)": KNNClassifier(k=5),
    "Naive Bayes": GaussianNB() 
}