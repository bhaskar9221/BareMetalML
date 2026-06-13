import sys
sys.path.append('/home/chotu/Projects/BareMetalML')
import numpy as np
from notebooks.naive_bayes.naive_bayes import GaussianNB, MultinomialNB

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


# feature with ZERO variance (all same value) for class 0
X_degenerate = np.array([
    [1.0, 5.0],
    [1.0, 6.0],
    [1.0, 7.0],
    [2.0, 1.0],
    [2.0, 2.0],
    [2.0, 3.0],
])
y_degenerate = np.array([0, 0, 0, 1, 1, 1])

# WITHOUT smoothing
nb_no_smooth = GaussianNB(var_smoothing=0)
nb_no_smooth.fit(X_degenerate, y_degenerate)
print("Var for class 0 (no smoothing):", nb_no_smooth.var_[0])  # [0. , something]

try:
    print(nb_no_smooth.predict(np.array([[1.0, 6.0]])))
except Exception as e:
    print("Error:", e)

# WITH smoothing
nb_smooth = GaussianNB(var_smoothing=1e-9)
nb_smooth.fit(X_degenerate, y_degenerate)
print("Var for class 0 (with smoothing):", nb_smooth.var_[0])
print("Prediction:", nb_smooth.predict(np.array([[1.0, 6.0]])))  # should work, predict class 0



# vocabulary: [free, money, meeting, project, win, schedule]
# each row = word counts for one message

X_text = np.array([
    [2, 1, 0, 0, 1, 0],   # "free money... win" -> spam
    [3, 2, 0, 0, 2, 0],   # spam
    [1, 1, 0, 0, 1, 0],   # spam
    [0, 0, 1, 1, 0, 1],   # "meeting project schedule" -> ham
    [0, 0, 2, 1, 0, 1],   # ham
    [0, 0, 1, 2, 0, 0],   # ham
])
y_text = np.array([1, 1, 1, 0, 0, 0])  # 1=spam, 0=ham

mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_text, y_text)

# test message: lots of "free" and "win" -> should predict spam (1)
test_spam = np.array([[3, 1, 0, 0, 2, 0]])
print("Spam test prediction:", mnb.predict(test_spam))  # expect [1]

# test message: lots of "meeting" and "schedule" -> should predict ham (0)
test_ham = np.array([[0, 0, 3, 1, 0, 2]])
print("Ham test prediction:", mnb.predict(test_ham))  # expect [0]