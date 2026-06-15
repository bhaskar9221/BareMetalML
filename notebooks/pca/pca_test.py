import sys
sys.path.append('/home/chotu/Projects/BareMetalML')
import numpy as np
from notebooks.pca.pca import PCA
from sklearn.datasets import load_iris
from utils import mean_squared_error




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


iris = load_iris()
X = iris.data   

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
X_reconstructed = pca.inverse_transform(X_reduced)

print("Original shape:", X.shape)          
print("Reduced shape:", X_reduced.shape)   
print("Reconstructed shape:", X_reconstructed.shape)  
print("Reconstruction MSE:", mean_squared_error(X, X_reconstructed))


pca_full = PCA(n_components=4)   
pca_full.fit(X)

print("Explained variance ratio per component:")
print(pca_full.explained_variance_ratio_)

cumulative = np.cumsum(pca_full.explained_variance_ratio_)
print("\nCumulative explained variance:")
print(cumulative)

n_for_95 = np.argmax(cumulative >= 0.95) + 1
print(f"\nComponents needed for 95% variance: {n_for_95}")


import matplotlib.pyplot as plt

iris = load_iris()
X, y = iris.data, iris.target

pca2 = PCA(n_components=2)
X_2d = pca2.fit_transform(X)

plt.figure(figsize=(7, 5))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='viridis', edgecolors='black')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Iris — PCA 2D")
plt.colorbar(scatter)
plt.savefig('pca_2d.png')
plt.close()

pca3 = PCA(n_components=3)
X_3d = pca3.fit_transform(X)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=y, cmap='viridis', edgecolors='black')
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("Iris — PCA 3D")
plt.savefig('pca_3d.png')
plt.close()

X_2feat = X[:, :2]
pca_2feat = PCA(n_components=2)
pca_2feat.fit(X_2feat)

plt.figure(figsize=(7, 5))
plt.scatter(X_2feat[:, 0], X_2feat[:, 1], c=y, cmap='viridis', alpha=0.5, edgecolors='black')

for i, (comp, var) in enumerate(zip(pca_2feat.components_, pca_2feat.explained_variance_)):
    arrow = comp * np.sqrt(var) * 2
    plt.arrow(pca_2feat.mean_[0], pca_2feat.mean_[1], 
              arrow[0], arrow[1], 
              color='red', width=0.02, head_width=0.15,
              label=f'PC{i+1}' if i == 0 else None)

plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title("Principal Components on Original Data")
plt.savefig('pca_arrows.png')
plt.close()

print("All 3 plots saved")