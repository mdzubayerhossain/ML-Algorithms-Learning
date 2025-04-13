import numpy as np

class PolynomialRegression:
    def __init__(self, degree=2):
        self.degree = degree
        self.model = LinearRegression()

    def fit(self, X, y):
        X_poly = self._polynomial_features(X)
        self.model.fit(X_poly, y)

    def predict(self, X):
        X_poly = self._polynomial_features(X)
        return self.model.predict(X_poly)

    def _polynomial_features(self, X):
        return np.column_stack([X**i for i in range(1, self.degree + 1)])
