import sys
sys.path.append('/home/chotu/Projects/BareMetalML')  
from utils import train_test_split, accuracy_score, StandardScaler
import numpy as np
from notebooks.knn.knn import euclidean_distance, manhattan_distance, KNNClassifier, KNNRegressor

X = np.array([[1, 2], [4, 6], [0, 0]])
x1 = np.array([0, 0])

print(euclidean_distance(x1, X))   # [2.236, 7.211, 0.0]
print(manhattan_distance(x1, X))   # [3, 10, 0]

# timing test
X_big = np.random.randn(1000, 50)
x1_big = np.random.randn(50)

import time
start = time.time()
euclidean_distance(x1_big, X_big)
print("Euclidean time:", time.time() - start)

start = time.time()
manhattan_distance(x1_big, X_big)
print("Manhattan time:", time.time() - start)



from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

knn = KNNClassifier(k=5)
knn.fit(X_train, y_train)
preds = knn.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))   # expect > 0.90

# sweep k
import matplotlib.pyplot as plt
accs = []
for k in range(1, 21):
    knn = KNNClassifier(k=k)
    knn.fit(X_train, y_train)
    accs.append(accuracy_score(y_test, knn.predict(X_test)))


# generate noisy sine wave
np.random.seed(42)
X_sine = np.linspace(0, 2*np.pi, 100).reshape(-1, 1)
y_sine = np.sin(X_sine).ravel() + np.random.normal(0, 0.1, 100)

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_sine, y_sine, random_state=42)

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))

for i, k in enumerate([1, 3, 10]):
    knn_r = KNNRegressor(k=k)
    knn_r.fit(X_train_s, y_train_s)
    
    # predict over a smooth range for plotting
    X_plot = np.linspace(0, 2*np.pi, 200).reshape(-1, 1)
    y_plot = knn_r.predict(X_plot)
    
    plt.subplot(1, 3, i+1)
    plt.scatter(X_train_s, y_train_s, s=10, alpha=0.4, label='train data')
    plt.plot(X_plot, y_plot, color='red', label=f'k={k}')
    plt.legend()
    plt.title(f"k={k}")

plt.tight_layout()
plt.savefig('knn_regression.png')
print("Saved plot")



from utils import plot_decision_boundary

iris   = load_iris()
X_2d   = iris.data[:, 2:]    # petal length + petal width
y      = iris.target

X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(X_2d, y, random_state=42)
scaler     = StandardScaler()
X_train_2d = scaler.fit_transform(X_train_2d)

for k in [1, 5, 15]:
    knn_2d = KNNClassifier(k=k)
    knn_2d.fit(X_train_2d, y_train_2d)
    plot_decision_boundary(knn_2d, X_train_2d, y_train_2d, title=f"KNN k={k}")
    plt.savefig(f'knn_boundary_k{k}.png')
    plt.close()