import sys
sys.path.append('/home/chotu/Projects/BareMetalML')
import numpy as np
from notebooks.pca.pca import PCA


X_simple = np.array([
    [1, 2],
    [2, 4],
    [3, 6],
    [4, 8],
])

pca = PCA(n_components=1)
pca.fit(X_simple)

print("Mean:", pca.mean_)
print("Component:", pca.components_)
print("Explained variance:", pca.explained_variance_)

