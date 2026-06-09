import sys
sys.path.append('/home/chotu/Projects/BareMetalML')  
import numpy as np
from sklearn.datasets import load_diabetes
from utils import train_test_split, r2_score, StandardScaler
import matplotlib.pyplot as plt



class LinearRegression:

    def __init__(self, solver='gd', learning_rate = 0.01, n_iterations = 1000, alpha = 0.0):
        self.solver = solver
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.loss_history = []
        self.alpha = alpha

    def _add_bias(self, X):
        ones = np.ones((X.shape[0], 1))
        return np.hstack((ones, X))

    def fit(self, X, y):
        if self.solver == 'normal':
            self._fit_normal(X, y)
        elif self.solver == 'gd':
            self._fit_gd(X, y)

    def _fit_normal(self, X, y):
        X_b = self._add_bias(X)
        self.weights = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        
    
    def predict(self, X):
        X_b = self._add_bias(X)
        return X_b.dot(self.weights)

    def _fit_gd(self, X, y):
        X_b = self._add_bias(X)
        n, d = X_b.shape
        self.weights = np.zeros(d)
        self.loss_history = []

        for i in range(self.n_iterations):
            y_hat = X_b @ self.weights
            error = y_hat - y
            grad  = (2 / n) * X_b.T @ error
            grad = (2 / n) * X_b.T @ error
            grad = np.clip(grad, -1e10, 1e10)   

            if self.alpha > 0:
                reg    = 2 * self.alpha * self.weights.copy()
                reg[0] = 0.0
                grad  += reg

            self.weights -= self.learning_rate * grad
            self.loss_history.append(np.mean(error ** 2))


















####Testing####

data = load_diabetes()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

model = LinearRegression(solver='normal')
model.fit(X_train, y_train)

from sklearn.linear_model import LinearRegression as SklearnLR
sk_model = SklearnLR()
sk_model.fit(X_train, y_train)
sk_preds = sk_model.predict(X_test)
your_preds = model.predict(X_test)

print("Sklearn R²:", r2_score(y_test, sk_preds))
print("Your R²   :", r2_score(y_test, your_preds))
print("Max diff  :", np.max(np.abs(sk_preds - your_preds)))

model_gd = LinearRegression(solver='gd', learning_rate=0.1, n_iterations=2000)
model_gd.fit(X_train, y_train)
print("Normal R²:", r2_score(y_test, model.predict(X_test)))
print("GD     R²:", r2_score(y_test, model_gd.predict(X_test)))

plt.plot(model_gd.loss_history)
plt.xlabel("Iteration")
plt.ylabel("MSE Loss")
plt.title("Gradient Descent Convergence")
plt.show()

alphas = [0.0, 0.1, 1.0, 10.0, 100.0]
for a in alphas:
    m = LinearRegression(solver='gd', learning_rate=0.01, n_iterations=2000, alpha=a)
    m.fit(X_train, y_train)
    w_norm = np.sum(m.weights[1:] ** 2)
    print(f"alpha={a:6.1f} | R²={r2_score(y_test, m.predict(X_test)):.4f} | weight norm={w_norm:.4f}")

# ── Residual Plot ─────────────────────────────── ← ALWAYS LAST
your_preds = model.predict(X_test)
residuals  = y_test - your_preds
plt.figure(figsize=(8, 4))
plt.scatter(your_preds, residuals, edgecolors='black', alpha=0.6, s=40)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot — Linear Regression")
plt.tight_layout()
plt.show()