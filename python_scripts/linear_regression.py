import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt


class LinearRegression:

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.losses = []

    def fit(self, X, y):
        '''Takes training samples and performs gradient descent'''

        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent
        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias
            self.losses.append(self.mse(y, y_predicted))

            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        '''Return the predicted value for a new data sample'''
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted

    def mse(self, y_true, y_predicted):
        return np.mean((y_true - y_predicted) ** 2)


if __name__ == "__main__":
    X, y = datasets.make_regression(
        n_samples=100, n_features=1, noise=20, random_state=4)

    # 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234)

    regressor = LinearRegression(lr=0.1)
    regressor.fit(X_train, y_train)
    predicted = regressor.predict(X_test)
    mse_value = regressor.mse(y_test, predicted)
    print(f"Final Training Loss Value: {regressor.losses[-1]}")
    print(f"Final Validation Loss Value: {mse_value}")

    # Visualize results
    plt.figure(1)
    pred_line = regressor.predict(X)
    cmap = plt.get_cmap("viridis")
    m1 = plt.scatter(X_train, y_train, color=cmap(0.8), s=10)
    m1 = plt.scatter(X_test, y_test, color=cmap(0.2), s=10)
    plt.plot(X, pred_line, color="black", linewidth=2, label="Prediction")
    plt.title("Linear Regression")
    plt.xlabel("x")
    plt.ylabel("y")
    legend = plt.legend(loc="lower right")

    # Plot Loss
    n_iters = 1000
    plt.figure(2)
    plt.plot(np.arange(0, n_iters), regressor.losses, label="Training Loss")
    plt.title("Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("MSE Value")
    legend2 = plt.legend(loc="upper right")

    plt.show()
