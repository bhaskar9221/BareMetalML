import sys
sys.path.append('/home/chotu/Projects/BareMetalML')
from linear_regression import LinearRegression
from utils import train_test_split, r2_score, StandardScaler



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