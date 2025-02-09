import numpy as np
from scipy.spatial import distance

class ServerKNNreg:
    def __init__(self, k=3, clients=None, distance_function=distance.euclidean):
        self.k = k
        self.distance_function = distance_function
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
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [k_nearest_labels[i] for i in k_indices]
        y_pred = np.average(k_nearest_labels)
        return y_pred

class ClientKNNreg:
    def __init__(self, k=3, distance_function=distance.euclidean):
        self.k = k
        self.distance_function = distance_function

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def fed_predict(self, X_test):
        predictions = self._predict(X_test)
        return np.array(predictions)

    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = [self.distance_function(x, x_train) for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        k_distances = [distances[i] for i in k_indices]
        return k_distances, k_nearest_labels


class KNNreg:
    def __init__(self, k=3, distance_function=distance.euclidean):
        self.k = k
        self.distance_function = distance_function

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            distances = [self.distance_function(x, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            # k_distances = [distances[i] for i in k_indices]
            k_labels = [self.y_train[i] for i in k_indices]
            y_pred = np.average(k_labels)
            predictions.append(y_pred)
        return np.array(predictions)
