import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
np.random.seed(1234)


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))


class KMeans:

    def __init__(self, K=5, max_iters=100, plot_steps=False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        self.clusters = [[] for _ in range(self.K)]
        self.centroids = []

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        # Random initialization
        sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in sample_idxs]

        for _ in range(self.max_iters):
            # Cluster assignment
            self.clusters = self._create_clusters(self.centroids)

            if self.plot_steps:
                self.plot()

            # Move centroids
            old_centroids = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            # Check for convergence; If algorithm converged, break
            if self._is_converged(old_centroids, self.centroids):
                break

            if self.plot_steps:
                self.plot(converged=False)

        # Classify samples as index of their clusters
        return self._get_cluster_labels(self.clusters)

    def _get_cluster_labels(self, clusters):
        '''Assign samples the cluster labels they were given'''
        labels = np.empty(self.n_samples)

        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels

    def _create_clusters(self, centroids):
        '''Assign samples to closest centroids'''
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        '''Calculate distance of each sample to closest centroid'''
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx

    def _get_centroids(self, clusters):
        '''Assign mean value of clusters to centroids'''
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, old_centroids, centroids):
        '''Calculates distances between old and new centroids for all centroids'''
        distances = [euclidean_distance(
            old_centroids[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0

    def plot(self, converged=False):
        '''Visualize results'''
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)

        plt.xlabel("x_1")
        plt.ylabel("x_2")

        if converged:
            print("K-Means has converged.")
            plt.title("K-Means Converged")
        else:
            print("K-Means has not converged.")
            plt.title("K-Means Not Converged")

        plt.show()


if __name__ == "__main__":
    X, y = make_blobs(centers=3, n_samples=500, n_features=2,
                      shuffle=True, random_state=1234)

    clusters = len(np.unique(y))
    print(f"Number of clusters: {clusters}")
    k = KMeans(K=clusters, max_iters=150, plot_steps=True)
    y_pred = k.predict(X)

    k.plot(converged=True)
