import numpy as np
from scipy.spatial import distance

class ServerFuzzyKNN:
    def __init__(self, k=3, m=2, clients=None):
        self.k = k
        self.m = m
        if clients is not None:
            self.clients = clients
        else:
            self.clients = []

    def add_clients(self,clients):
        self.clients = clients

    def predict(self, x: list[list]):
        # x = [[2,2],[3,3]]
        # x = [[2,2]]
        distances = [[] for _ in x]
        for client in self.clients:
            distance_class = client.predict_fed(x)
            for i in range(len(x)):
                distances[i].extend(distance_class[i])
        return self.predict_fed(distances)

    def predict_fed(self, dist_class_pairs_list):
        predictions = []
        for x in dist_class_pairs_list:
            memberships = self.compute_membership_from_pairs(x)
            predicted_class = self._defuzzify(memberships)
            predictions.append(predicted_class)
        return np.array(predictions)

    def compute_membership_from_pairs(self, distance_class_triplets):
        memberships = {}
        classes = np.unique([x[1] for x in distance_class_triplets])
        for label in classes:
            int_label = int(label)
            memberships[int_label] = 0
            membership_u = 0
            membership = 0
            for dist, cls, ms in distance_class_triplets:
                if cls == int_label:
                    membership_u += ms / (dist ** (2 / (self.m - 1)) + 1e-5)
                    membership += 1 / (dist ** (2 / (self.m - 1)) + 1e-5)
            memberships[int_label] = membership_u / (membership + 1e-5)
        return memberships

    def _defuzzify(self, memberships):
        return max(memberships, key=memberships.get)

class CliFuzzyKNN:
    def __init__(self, k=3, m=2):
        self.k = k
        self.kInit = k
        self.m = m

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.classes = np.unique(y_train)
        self.memberships = self._compute_memberships(X_train, y_train)

    def _compute_memberships(self, X_train, y_train):
        memberships = []
        for i, xi in enumerate(X_train):
            distances = [distance.euclidean(xi, x_train) for x_train in X_train]
            kInit_indices = np.argsort(distances)[1:self.kInit+1]  # Exclude the instance itself
            kInit_labels = [y_train[j] for j in kInit_indices]
            instance_memberships = {}
            for c in self.classes:
                vc = kInit_labels.count(c)
                if c == y_train[i]:
                    instance_memberships[c] = 0.51 + (vc / self.kInit) * 0.49
                else:
                    instance_memberships[c] = (vc / self.kInit) * 0.49
            memberships.append(instance_memberships)
        return memberships

    def predict_fed(self, X_test):
        predictions = []
        for x in X_test:
            distances = [distance.euclidean(x, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_distances = [distances[i] for i in k_indices]
            k_labels = [self.y_train[i] for i in k_indices]
            k_memberships = [self.memberships[i][l] for i,l in zip(k_indices,k_labels)]
            predictions.append(list(zip([float(x) for x in k_distances], [int(x) for x in k_labels], [x for x in k_memberships])))
        return predictions

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            distances = [distance.euclidean(x, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_distances = [distances[i] for i in k_indices]
            k_labels = [self.y_train[i] for i in k_indices]
            k_memberships = [self.memberships[i][l] for i, l in zip(k_indices, k_labels)]
            memberships = self._calculate_memberships(k_distances, k_labels, k_memberships)
            predicted_class = self._defuzzify(memberships)
            predictions.append(predicted_class)
        return np.array(predictions)

    def _calculate_memberships(self, k_distances, k_labels, k_memberships):
        memberships = {}
        for label in self.classes:
            memberships[label] = 0
            membership_u = 0
            membership = 0
            for i in range(self.k):
                if k_labels[i] == label:
                    membership_u += k_memberships[i] / (k_distances[i] ** (2 / (self.m - 1)) + 1e-5)
                    membership += 1 / (k_distances[i] ** (2 / (self.m - 1)) + 1e-5)
            memberships[label] = membership_u / (membership + 1e-5)
        return memberships

    def compute_membership_from_pairs(self, distance_class_triplets):
        memberships = {}
        for label in self.classes:
            int_label = int(label)
            memberships[int_label] = 0
            membership_u = 0
            membership = 0
            for dist, cls, ms in distance_class_triplets:
                if cls == int_label:
                    membership_u += ms / (dist ** (2 / (self.m - 1)) + 1e-5)
                    membership += 1 / (dist ** (2 / (self.m - 1)) + 1e-5)
            memberships[int_label] = membership_u / (membership + 1e-5)
        return memberships

    def _defuzzify(self, memberships):
        return max(memberships, key=memberships.get)


class FuzzyKNN:
    def __init__(self, k=3, m=2):
        self.k = k
        self.kInit = k
        self.m = m

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.classes = np.unique(y_train)
        self.memberships = self._compute_memberships(X_train, y_train)

    def _compute_memberships(self, X_train, y_train):
        memberships = []
        for i, xi in enumerate(X_train):
            distances = [distance.euclidean(xi, x_train) for x_train in X_train]
            kInit_indices = np.argsort(distances)[1:self.kInit+1]  # Exclude the instance itself
            kInit_labels = [y_train[j] for j in kInit_indices]
            instance_memberships = {}
            for c in self.classes:
                vc = kInit_labels.count(c)
                if c == y_train[i]:
                    instance_memberships[c] = 0.51 + (vc / self.kInit) * 0.49
                else:
                    instance_memberships[c] = (vc / self.kInit) * 0.49
            memberships.append(instance_memberships)
        return memberships


    def predict(self, X_test):
        predictions = []
        for x in X_test:
            distances = [distance.euclidean(x, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_distances = [distances[i] for i in k_indices]
            k_labels = [self.y_train[i] for i in k_indices]
            k_memberships = [self.memberships[i][l] for i, l in zip(k_indices, k_labels)]
            memberships = self._calculate_memberships(k_distances, k_labels, k_memberships)
            predicted_class = self._defuzzify(memberships)
            predictions.append(predicted_class)
        return np.array(predictions)

    def _calculate_memberships(self, k_distances, k_labels, k_memberships):
        memberships = {}
        for label in self.classes:
            memberships[label] = 0
            membership_u = 0
            membership = 0
            for i in range(self.k):
                if k_labels[i] == label:
                    membership_u += k_memberships[i] / (k_distances[i] ** (2 / (self.m - 1)) + 1e-5)
                    membership += 1 / (k_distances[i] ** (2 / (self.m - 1)) + 1e-5)
            memberships[label] = membership_u / (membership + 1e-5)
        return memberships

    def _defuzzify(self, memberships):
        return max(memberships, key=memberships.get)
