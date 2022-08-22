from typing import Union

import numpy as np

from .utils import distance


class SimpleNNClassifier:
    """
    Optimization moment of this algorithm is the find point with the minimal distance.
    Instead of O(N* log(N)) complexity for sorting,
    the complexity of finding the minimum element is O(N).
    """
    def __init__(self):
        self._X_train = None
        self._y_train = None

    def fit(self, X: np.array, y: Union[np.array, list]) -> None:
        """
        In a simple version of the algorithm, there is no need for additional operations
        other than saving the training set.
        :param X:
        :param y:
        :return:
        """
        if X.shape[0] != len(y):
            raise ValueError("X and y lengths don't fit")

        self._X_train = X
        self._y_train = y

    def predict(self, X: np.array) -> list:
        distances = []
        for x_predict_point in X:
            min_distance, label = np.inf, None
            for x_train_point, point_label in zip(self._X_train, self._y_train):
                point_distance = distance(x_train_point, x_predict_point)
                if point_distance < min_distance:
                    min_distance, label = point_distance, point_label

            distances.append((min_distance, label))
        # 1 - is the index of label in tuple
        y_predicted = [point[1] for point in distances]

        return y_predicted
