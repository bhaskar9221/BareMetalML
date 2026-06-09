import numpy as np
import sys
sys.path.append('/home/chotu/Projects/BareMetalML')


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






####Testing####

# sigmoid
print(sigmoid(0))      
print(sigmoid(100))    
print(sigmoid(-100))   

# BCE
y_true = np.array([1, 0, 1])
y_pred_perfect = np.array([0.999, 0.001, 0.999])
y_pred_wrong   = np.array([0.001, 0.999, 0.001])
print(_binary_cross_entropy(y_true, y_pred_perfect))  
print(_binary_cross_entropy(y_true, y_pred_wrong))        



from sklearn.datasets import load_iris
from utils import train_test_split, accuracy_score, StandardScaler

iris = load_iris()
X, y = iris.data, iris.target

mask = y < 2
X, y = X[mask], y[mask]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
model.fit(X_train, y_train)

print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))  
print("Loss history length:", len(model.loss_history))             
print("First loss:", round(model.loss_history[0], 4))              
print("Last loss: ", round(model.loss_history[-1], 4))             



iris = load_iris()
X, y = iris.data, iris.target        # full 3-class this time

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

ovr = OneVsRestClassifier(learning_rate=0.1, n_iterations=1000)
ovr.fit(X_train, y_train)
preds = ovr.predict(X_test)

from utils import accuracy_score, confusion_matrix
print("Accuracy:", accuracy_score(y_test, preds))       # expect > 0.90
print("Confusion Matrix:\n", confusion_matrix(y_test, preds))