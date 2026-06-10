
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