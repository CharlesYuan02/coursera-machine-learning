import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d


class LogisticRegression:

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.accuracies = []

    def fit(self, X, y):
        '''Takes training samples and performs gradient descent'''

        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent
        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias

            # Pass linear regression output through sigmoid
            y_predicted = self._sigmoid(y_predicted)
            self.accuracies.append(self.accuracy(y, y_predicted))

            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        '''Return the predicted value for a new data sample'''
        y_predicted = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(y_predicted)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_cls

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def accuracy(self, y_true, y_predicted):
        return np.sum(y_true == y_predicted) / len(y_true)


if __name__ == "__main__":
    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target

    # 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234)

    regressor = LogisticRegression(lr=0.1, n_iters=1000)
    regressor.fit(X_train, y_train)
    predicted = regressor.predict(X_test)
    accuracy = regressor.accuracy(y_test, predicted)
    print(f"Final Training Accuracy: {regressor.accuracies[-1]}")
    print(f"Final Validation Accuracy: {accuracy}")
 
    # Note that the shape of the matrix will not allow you to plot a 2D line like linear regression
    print(regressor.weights, regressor.bias)

    # Visualize results
    plt.figure(1)
    cmap = plt.get_cmap("viridis")
    m1 = plt.scatter(X_train[0], X_train[1], color=cmap(0.8), s=10)
    m1 = plt.scatter(X_test[0], X_test[1], color=cmap(0.2), s=10)
    plt.title("Breast Cancer Data")
    plt.xlabel("x")
    plt.ylabel("y")

    # Plot Accuracy
    n_iters = 1000
    plt.figure(2)
    accuracies_smoothed = gaussian_filter1d(regressor.accuracies, sigma=3)
    plt.plot(np.arange(0, n_iters), accuracies_smoothed,
             label="Training Accuracy")
    plt.title("Training Accuracy")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    legend2 = plt.legend(loc="upper right")

    plt.show()
