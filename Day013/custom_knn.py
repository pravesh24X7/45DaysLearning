# This file contains the code segment for the CustomKNN class

import numpy as np

class CustomKNN:
    def __init__(self, n_neighbours=5):
        self.n_neighbours = n_neighbours

    def fit(self, X_train, y_train):
        self.X, self.y = X_train, y_train

    def __distance(self, p1, p2):
        return np.sqrt(
            np.dot(p1-p2, p1-p2)
        )

    def predict(self, X_test):

        y_pred = []
        for test in X_test:
            all_distances = []
            for train_idx in range( self.X.shape[0] ):
                distance = self.__distance(self.X[train_idx], test)
                all_distances.append(
                    ( train_idx, distance )
                )

            all_distances.sort(key=lambda item: item[1], reverse=False)
            top_k = all_distances[: self.n_neighbours]
            labels = self.y[ [ idx[0] for idx in top_k ] ]

            most_frequent_label = np.argmax( np.bincount(labels) )
            y_pred.append(most_frequent_label)

        return np.array(y_pred)