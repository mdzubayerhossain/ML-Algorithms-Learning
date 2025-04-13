import numpy as np

class KMeans:
    def __init__(self, k=2, max_iters=100):
        self.k = k
        self.max_iters = max_iters

    def fit(self, X):
        self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]
        for _ in range(self.max_iters):
            labels = [self._closest_centroid(x) for x in X]
            new_centroids = [np.mean(X[np.array(labels) == i], axis=0) for i in range(self.k)]
            self.centroids = np.array(new_centroids)

    def _closest_centroid(self, x):
        return np.argmin([np.linalg.norm(x - point) for point in self.centroids])
