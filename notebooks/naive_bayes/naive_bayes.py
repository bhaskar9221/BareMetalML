import numpy as np
import sys
sys.path.append('/home/chotu/Projects/BareMetalML')


class GaussianNB:
    def __init__(self):
        self.classes_ = None 
        self.mean_ = None 
        self.var_ = None 
        self.priors_ = None
    
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.mean_ = {}
        self.var_ = {}
        self.priors_ = {}

        for c in self.classes_:
            X_c = X[y == c]
            self.mean_[c] = np.mean(X_c, axis=0)
            self.var_[c] = np.var(X_c, axis=0)
            self.priors_[c] = len(X_c) / len(X)
    
    def _gaussian_log_pdf(self, x, mean, var):
        return -0.5 * np.log(2 * np.pi * var) - ((x - mean) ** 2) / (2 * var)
    
    def _predict_one(self, x):
        posteriors = []

        for c in self.classes_:
            prior = np.log(self.priors_[c])
            likelihood = np.sum(
                self._gaussian_log_pdf(x, self.mean_[c], self.var_[c])
            )
            posteriors.append(prior + likelihood)

        return self.classes_[np.argmax(posteriors)]
    
    def predict(self, X):
        return np.array([self._predict_one(x) for x in X])








from sklearn.datasets import load_iris
from utils import train_test_split, accuracy_score, StandardScaler

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

nb = GaussianNB()
nb.fit(X_train, y_train)
preds = nb.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))   # expect > 0.90

print("\nLearned means per class:")
for c in nb.classes_:
    print(f"  Class {c}: {nb.mean_[c].round(2)}")