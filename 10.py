import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Load the Iris dataset
X, y = load_iris(return_X_y=True)

# Perform PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Perform LDA
lda = LDA(n_components=2)
X_lda = lda.fit_transform(X, y)

print("Shape of Data:", X.shape)
print("Shape of PCA transformed Data:", X_pca.shape)
print("Shape of LDA transformed Data:", X_lda.shape)

plt.figure(figsize=(12, 6))

# Plot PCA results
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="jet")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of Iris Dataset (sklearn)")


# Plot LDA results
plt.subplot(1, 2, 2)
plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap="jet")
plt.xlabel("Linear Discriminant 1")
plt.ylabel("Linear Discriminant 2")
plt.title("LDA of Iris Dataset (sklearn)")

plt.tight_layout()
plt.show()