# k_nearest_neighbors.py: Machine learning implementation of a K-Nearest Neighbors classifier from scratch.


import numpy as np
from utils import euclidean_distance, manhattan_distance


class KNearestNeighbors:
    """
    A class representing the machine learning implementation of a K-Nearest Neighbors classifier from scratch.

    Attributes:
        n_neighbors
            An integer representing the number of neighbors a sample is compared with when predicting target class
            values.

        weights
            A string representing the weight function used when predicting target class values. The possible options are
            {'uniform', 'distance'}.

        _X
            A numpy array of shape (n_samples, n_features) representing the input data used when fitting the model and
            predicting target class values.

        _y
            A numpy array of shape (n_samples,) representing the true class values for each sample in the input data
            used when fitting the model and predicting target class values.

        _distance
            An attribute representing which distance metric is used to calculate distances between samples. This is set
            when creating the object to either the euclidean_distance or manhattan_distance functions defined in
            utils.py based on what argument is passed into the metric parameter of the class.

    Methods:
        fit(X, y)
            Fits the model to the provided data matrix X and targets y.

        predict(X)
            Predicts class target values for the given test data matrix X using the fitted classifier model.
    """

    def __init__(self, n_neighbors = 5, weights = 'uniform', metric = 'l2'):
        # Check if the provided arguments are valid
        if weights not in ['uniform', 'distance'] or metric not in ['l1', 'l2'] or not isinstance(n_neighbors, int):
            raise ValueError('The provided class parameter arguments are not recognized.')

        # Define and setup the attributes for the KNearestNeighbors model object
        self.n_neighbors = n_neighbors
        self.weights = weights
        self._X = None
        self._y = None
        self._distance = euclidean_distance if metric == 'l2' else manhattan_distance

    def fit(self, X, y):
        """
        Fits the model to the provided data matrix X and targets y.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.
            y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data.

        Returns:
            None.
        """
        self._X = X
        self._y = y

    def predict(self, X):
        """
        Predicts class target values for the given test data matrix X using the fitted classifier model.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the test data.

        Returns:
            A numpy array of shape (n_samples,) representing the predicted target class values for the given test data.
        """
        # Check if fit has been called
        if self._X is None or self._y is None:
            raise ValueError("This KNearestNeighbors instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        # List to hold the predicted class values
        predictions = []

        # Iterate over each test point
        for test_point in X:
            # Calculate the distance between the test point and all training points
            distances = np.array([self._distance(test_point, train_point) for train_point in self._X])

            # Get the indices of the k nearest neighbors
            neighbor_indices = distances.argsort()[:self.n_neighbors]

            # Get the labels of the k nearest neighbors
            neighbor_labels = self._y[neighbor_indices]

            if self.weights == 'uniform':
                # Use majority vote of the neighbor labels as the prediction
                prediction = np.argmax(np.bincount(neighbor_labels))
            else:
                # Weighted vote: neighbors closer to the test point have more weight in the voting
                # Calculate weights (inverse of distance)
                neighbor_distances = distances[neighbor_indices]
                # Handle case to avoid division by zero
                neighbor_distances = np.where(neighbor_distances == 0, 1e-5, neighbor_distances)
                weights = 1 / neighbor_distances
                weighted_vote = np.bincount(neighbor_labels, weights=weights)
                prediction = np.argmax(weighted_vote)

            # Append prediction to the list
            predictions.append(prediction)

        return np.array(predictions)

