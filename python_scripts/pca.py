import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


class PCA:

    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # Mean normalization
        self.mean = np.mean(X, axis=0)
        X -= self.mean

        # Compute covariance matrix
        cov = np.cov(X.T)

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        eigenvectors = eigenvectors.T  # Transpose for easier calculations

        # Sorts eigenvalues, reverses order so that they're [0 1 2 3] instead of [3 2 1 0]
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        # Store first n eigenvectors
        self.components = eigenvectors[0:self.n_components]

    def transform(self, X):
        '''Dimension reduction'''
        X -= self.mean
        return np.dot(X, self.components.T)


if __name__ == "__main__":
    data = datasets.load_iris()
    X = data.data
    y = data.target

    # Project data onto 2 components (i.e. Make data 2D)
    pca = PCA(2)
    pca.fit(X)
    X_projected = pca.transform(X)

    print(f"Shape of X: {X.shape}")
    print(f"Shape of dimension reduced X: {X_projected.shape}")

    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]

    plt.scatter(x1, x2, c=y, edgecolor="none", alpha=0.8,
                cmap=plt.cm.get_cmap("viridis", 3))

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar()
    plt.show()
