import numpy as np
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))


class KNN:

    def __init__(self, K=3):
        self.K = K

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # Compute distances between data point x and all examples in training set
        distances = [euclidean_distance(x, x_train)
                     for x_train in self.X_train]

        # Sort by distance and return indices of the first K neighbours
        K_idx = np.argsort(distances)[:self.K]

        # Extract labels of the K nearest neighbour training samples
        K_neighbour_labels = [self.y_train[i] for i in K_idx]

        # Return the most common class label
        most_common = Counter(K_neighbour_labels).most_common(1)
        return most_common[0][0]

    def accuracy(self, y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        self.acc = accuracy
        return accuracy


if __name__ == "__main__":
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234)

    plt.figure(1)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test,
                cmap=cmap, edgecolor="k", s=20)
    plt.title("KNN Validation Data True Labels")
    plt.xlabel("x_1")
    plt.ylabel("x_2")

    K = 3
    clf = KNN(K=K)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    plt.figure(2)
    for i in range(len(X_test)):
        if predictions[i] == 0:
            plt.scatter(X_test[i, 0], X_test[i, 1], marker="o",
                        color="red", edgecolor="k", s=20)
        elif predictions[i] == 1:
            plt.scatter(X_test[i, 0], X_test[i, 1], marker="o",
                        color="green", edgecolor="k", s=20)
        elif predictions[i] == 2:
            plt.scatter(X_test[i, 0], X_test[i, 1], marker="o",
                        color="blue", edgecolor="k", s=20)
        else:
            print("Please modify code for K>3")

    print(f"KNN Accuracy: {clf.accuracy(y_test, predictions)*100}%")
    plt.title("KNN Validation Data Predicted Labels")
    plt.xlabel("x_1")
    plt.ylabel("x_2")

    # You can see the accuracy is 100% and the graphs are equivalent
    plt.show()
