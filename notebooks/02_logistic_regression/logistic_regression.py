import numpy as np
import sys
sys.path.append('/home/chotu/Projects/BareMetalML')
from sklearn.datasets import load_iris
from utils import train_test_split, StandardScaler

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def _binary_cross_entropy(y_true, y_pred):
    z = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(y_true * np.log(z) + (1 - y_true) * np.log(1 - z))




class LogisticRegression:
    def __init__(self,learning_rate = 0.01, n_iterations = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.loss_history = []

    def _add_bias(self, X):
        ones = np.ones((X.shape[0], 1))
        return np.hstack((ones, X))

    def fit(self, X, y):
        X_b     = self._add_bias(X)
        n, d    = X_b.shape
        self.weights      = np.zeros(d)
        self.loss_history = []

        for _ in range(self.n_iterations):
            z     = X_b @ self.weights        
            y_hat = sigmoid(z)                
            error = y_hat - y                 
            grad  = (1 / n) * X_b.T @ error  
            self.weights -= self.learning_rate * grad
            self.loss_history.append(_binary_cross_entropy(y, y_hat))

    def predict_proba(self, X):
        X_b = self._add_bias(X)
        return sigmoid(X_b @ self.weights)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


class OneVsRestClassifier:
    
    def __init__(self, learning_rate=0.1, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations  = n_iterations
        self.models        = []   

    def fit(self, X, y):
        self.classes_ = np.unique(y)          
        self.models   = []

        for c in self.classes_:
            y_binary = (y == c).astype(int)
            model = LogisticRegression(
                learning_rate=self.learning_rate,
                n_iterations=self.n_iterations
            )
            model.fit(X, y_binary)
            self.models.append(model)

    def predict(self, X):
        probs = np.array([m.predict_proba(X) for m in self.models])
        return np.argmax(probs, axis=0)

from utils import plot_decision_boundary

iris   = load_iris()
X_2d   = iris.data[:, 2:]    
y      = iris.target

X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
    X_2d, y, random_state=42
)
scaler    = StandardScaler()
X_train_2d = scaler.fit_transform(X_train_2d)
X_test_2d  = scaler.transform(X_test_2d)

ovr_2d = OneVsRestClassifier(learning_rate=0.1, n_iterations=1000)
ovr_2d.fit(X_train_2d, y_train_2d)

plot_decision_boundary(
    ovr_2d, X_train_2d, y_train_2d,
    title="Logistic Regression — One vs Rest (Iris)"
)







