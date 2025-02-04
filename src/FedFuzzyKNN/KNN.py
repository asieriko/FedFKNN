import numpy as np
from networkx.algorithms.connectivity import k_edge_subgraphs
from scipy.spatial import distance
from collections import Counter

class ServerKNN:
    def __init__(self, k=3, clients=None):
        self.k = k
        self.clients = clients

    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)

    def _predict(self, x):
        distances = []
        k_nearest_labels = []
        for client in self.clients:
            k_distances_i, k_nearest_labels_i = client.fed_predict(x)
            distances.extend(k_distances_i)
            k_nearest_labels.extend(k_nearest_labels_i)
        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [k_nearest_labels[i] for i in k_indices]
        # Perform a majority vote and return the most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

class ClientKNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def fed_predict(self, X_test):
        predictions = self._predict(X_test)
        return np.array(predictions)

    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = [distance.euclidean(x, x_train) for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        k_distances = [distances[i] for i in k_indices]
        return k_distances, k_nearest_labels


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)

    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = [distance.euclidean(x, x_train) for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Perform a majority vote and return the most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# Example usage
if __name__ == "__main__":
    # Sample data
    X_train = np.array([[1, 2], [2, 3], [3, 4], [6, 7], [7, 8], [8, 9]])
    y_train = np.array([0, 0, 0, 1, 1, 1])
    X_test = np.array([[2, 2], [7, 7]])

    # Create and train the model
    knn = KNN(k=3)
    knn.fit(X_train, y_train)

    # Make predictions
    predictions = knn.predict(X_test)
    print("Predictions:", predictions)