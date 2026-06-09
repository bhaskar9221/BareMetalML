import numpy as np
import sys
sys.path.append('/home/chotu/Projects/BareMetalML')


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def _binary_cross_entropy(y_true, y_pred):
    z = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(y_true * np.log(z) + (1 - y_true) * np.log(1 - z))



####Testing####

# sigmoid
print(sigmoid(0))      # exactly 0.5
print(sigmoid(100))    # ~1.0
print(sigmoid(-100))   # ~0.0

# BCE
y_true = np.array([1, 0, 1])
y_pred_perfect = np.array([0.999, 0.001, 0.999])
y_pred_wrong   = np.array([0.001, 0.999, 0.001])
print(_binary_cross_entropy(y_true, y_pred_perfect))  # very small ~0.001
print(_binary_cross_entropy(y_true, y_pred_wrong))    # very large ~7.0    