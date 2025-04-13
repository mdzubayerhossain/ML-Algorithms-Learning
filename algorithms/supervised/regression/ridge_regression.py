import numpy as np

class RidgeRegression:
    def __init__(self, alpha=1):
        self.alpha = alpha
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        X_b = np.c_[np.ones((n_samples, 1)), X]
        A = X_b.T.dot(X_b) + self.alpha * np.identity(n_features + 1)
        b = X_b.T.dot(y)
        theta_best = np.linalg.inv(A).dot(b)
        self.bias = theta_best[0]
        self.weights = theta_best[1:]

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
