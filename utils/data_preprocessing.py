import numpy as np
from sklearn.model_selection import train_test_split

def handle_missing_values(X):
    return np.nan_to_num(X)

def split_data(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=42)
