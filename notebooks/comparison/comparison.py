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

results = []

for name, model in models.items():
    start = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - start

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc  = accuracy_score(y_test, model.predict(X_test))

    results.append({
        "Model": name,
        "Train Acc": round(train_acc, 4),
        "Test Acc": round(test_acc, 4),
        "Fit Time (s)": round(fit_time, 4),
    })


print(f"{'Model':<22} {'Train Acc':<12} {'Test Acc':<12} {'Fit Time (s)':<12}")
print("-" * 60)
for r in results:
    print(f"{r['Model']:<22} {r['Train Acc']:<12} {r['Test Acc']:<12} {r['Fit Time (s)']:<12}")


X_2d = iris.data[:, 2:]   
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(X_2d, y, random_state=42)
scaler_2d = StandardScaler()
X_train_2d = scaler_2d.fit_transform(X_train_2d)

models_2d = {
    "Logistic Regression": OneVsRestClassifier(learning_rate=0.1, n_iterations=1000),
    "Decision Tree":       DecisionTree(max_depth=5),
    "Random Forest":       RandomForest(n_trees=10, max_depth=5),
    "KNN (k=5)":           KNNClassifier(k=5),
    "Naive Bayes":         GaussianNB(),
}

fig, axes = plt.subplots(1, 5, figsize=(25, 5))

for ax, (name, model) in zip(axes, models_2d.items()):
    model.fit(X_train_2d, y_train_2d)

    x1_min, x1_max = X_train_2d[:, 0].min()-1, X_train_2d[:, 0].max()+1
    x2_min, x2_max = X_train_2d[:, 1].min()-1, X_train_2d[:, 1].max()+1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.05),
                            np.arange(x2_min, x2_max, 0.05))
    grid = np.c_[xx1.ravel(), xx2.ravel()]
    Z = model.predict(grid).reshape(xx1.shape)

    ax.contourf(xx1, xx2, Z, alpha=0.3, cmap='RdYlBu')
    ax.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train_2d, cmap='RdYlBu', edgecolors='black', s=20)
    ax.set_title(name)

plt.tight_layout()
plt.savefig('comparison_boundaries.png')
print("Saved comparison boundaries")