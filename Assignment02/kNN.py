import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from collections import Counter


class KNNClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, k=5, metric="euclidean"):
        self.k = k
        self.metric = metric

    def fit(self, X, y=None):
        self.X_train = X
        self.y_train = y
        return self

    def _euclidean_distance(self, p):
        return np.sqrt(np.sum((self.X_train - p) ** 2, axis=1))

    def _manhattan_distance(self, p):
        return np.sum(np.abs(self.X_train - p), axis=1)
    
    def _chebyshev_distance(self, p):
        return np.amax(np.abs(self.X_train - p), axis=1)

    def predict(self, X):
        pred_labels = []
        for i, p in enumerate(X):
            #print("Current data point:", i+1)
            distances = None
            if self.metric == "euclidean":
                distances = self._euclidean_distance(p)
            elif self.metric == "manhattan":
                distances = self._manhattan_distance(p)
            elif self.metric == "chebyshev":
                distances = self._chebyshev_distance(p)
            neighbors = np.argsort(distances)[0:self.k]
            neighbors_labels = self.y_train[neighbors]
            pred_label = Counter(neighbors_labels).most_common(1)[0][0]
            pred_labels.append(pred_label)
        return pred_labels
