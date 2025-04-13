import numpy as np
from algorithms.supervised.regression.linear_regression import LinearRegression

def test_linear_regression():
    X = np.random.rand(100, 1)
    y = 3 * X + 2 + np.random.randn(100, 1)
    
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    
    assert predictions.shape == y.shape, "Test Failed!"
