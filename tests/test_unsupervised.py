import numpy as np
from algorithms.unsupervised.clustering.k_means import KMeans

def test_kmeans():
    X = np.random.rand(100, 2)
    model = KMeans(k=3)
    model.fit(X)
    
    assert model.centroids.shape == (3, 2), "Test Failed!"
