import numpy as np
from scipy.spatial import distance

class ServerFuzzyKNNreg:
    def __init__(self, k=3, m=2, distance_function=distance.euclidean,clients=None):
        self.k = k
        self.m = m
        self.distance_function = distance_function
        if clients is not None:
            self.clients = clients
        else:
            self.clients = []

    def add_clients(self,clients):
        self.clients = clients

    def predict(self, x: list[list]):
        # x = [[2,2],[3,3]]
        # x = [[2,2]]
        distances = []
        for client in self.clients:
            distance_class = client.predict_fed(x)
            distances.append(distance_class)
        transposed_matrices = list(zip(*distances))
        distances = [[item for sublist in x for item in sublist] for x in transposed_matrices ]
        return self.predict_fed(distances)

    def predict_fed(self, dist_class_pairs_list):
        predictions = []
        for x in dist_class_pairs_list:
            # para cada ejemplos que tiene [[d1,y1,w1],[d2,y2,w2],...,[dn,yn,wn]]
            # flattened = [item for sublist in x for item in sublist]
            sorted_data = sorted(x, key=lambda x: x[0])
            selected_data = np.array(sorted_data[:self.k])
            y_pred = np.sum([w * y for w, y in selected_data[:,1:]]) / np.sum(selected_data[:,-1])
            predictions.append(y_pred)
        return np.array(predictions)


class CliFuzzyKNNreg:
    def __init__(self, k=3, m=2, distance_function=distance.euclidean):
        self.k = k
        self.kInit = k
        self.m = m
        self.distance_function = distance_function

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict_fed(self, X_test):
        predictions = []
        for x in X_test:
            distances = [self.distance_function(x, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_distances = [distances[i] for i in k_indices]
            k_labels = [self.y_train[i] for i in k_indices]
            k_weights = [1/((1/(d+1e-10))**(2/(self.m-1))) for d in k_distances]
            predictions.append([list(item) for item in zip(k_distances,k_labels,k_weights)])
        return predictions

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            distances = [self.distance_function(x, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_distances = [distances[i] for i in k_indices]
            k_labels = [self.y_train[i] for i in k_indices]
            k_weights = [1/((1/(d+1e-10))**(2/(self.m-1))) for d in k_distances]
            y_pred = np.sum([w * y for w, y in zip(k_weights, k_labels)]) / np.sum(k_weights)
            predictions.append(y_pred)
        return np.array(predictions)



class FuzzyKNNreg:
    def __init__(self, k=3, m=2, distance_function=distance.euclidean):
        self.k = k
        self.kInit = k
        self.m = m
        self.distance_function = distance_function

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            distances = [self.distance_function(x, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_distances = [distances[i] for i in k_indices]
            k_labels = [self.y_train[i] for i in k_indices]
            k_weights = [1/((1/(d+1e-10))**(2/(self.m-1))) for d in k_distances]
            y_pred = np.sum([w * y for w, y in zip(k_weights, k_labels)]) / np.sum(k_weights)
            predictions.append(y_pred)
        return np.array(predictions)
