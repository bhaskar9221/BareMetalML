import numpy as np
from sklearn.datasets import load_diabetes, load_iris

class LinearRegression:
    def __init__(self, solver = 'gd', lr = 0.01, n_iterations = 1000, alpha = 0.0):
        """
        Parameters
        """
        self.solver = solver
        self.lr = lr
        self.n_ieterations = n_iterations
        self.alpha = alpha
        self.weights = None
        self.loss_history = []

    
    def _add_bias(self, X):
        ones = np.ones((X.shape[0], 1))
        return np.hstack((ones, X))
    
    def fit(self, X, y):
        if self.solver == 'normal':
            self._fit_normal(X,y)
        else:
            self._fit_gd(X,y)
    
    def _fit_normal(self, X, y):
        X_b = self._add_bias(X)                      
        self.weights = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y

    def _fit_gd(self, X, y):
        X_b = self._add_bias(X)
        n, d = X_b.shape
        self.weights = np.zeros(d)
        self.loss_history = []

        for _ in range(self.n_ieterations):
            y_hat = X_b @ self.weights
            error = y_hat - y

            grad = (2/n) * X_b.T @ error 

            if self.alpha > 0:
                reg = 2 * self.alpha * self.weights.copy()
                reg[0] = 0.0
                grad += reg

            self.weights -= self.lr * grad

            loss = np.mean(error ** 2)
            self.loss_history.append(loss)
    
    def predict(self, X):
        X_b = self._add_bias(X)
        return X_b @ self.weights
    







########Test the implementation########
from sklearn.datasets import load_diabetes
from utils import train_test_split, r2_score, mean_squared_error, StandardScaler

data = load_diabetes()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Part A — Normal equation
model = LinearRegression(solver='normal')
model.fit(X_train, y_train)
preds = model.predict(X_test)
print("R²  :", r2_score(y_test, preds))      # expect 0.48 - 0.56
print("MSE :", mean_squared_error(y_test, preds))