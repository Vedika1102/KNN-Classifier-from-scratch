# KNN-Classifier-from-scratch

This repository contains a Python implementation of a K-Nearest Neighbors (KNN) classifier from scratch. KNN is a simple but effective machine learning algorithm used for classification and regression tasks. In this implementation, we provide a basic KNN classifier that can be used for classification tasks.

Table of Contents:

Introduction

Parameters

Methods

# Introduction
K-Nearest Neighbors is a non-parametric, instance-based learning algorithm that works by finding the K-nearest data points in the training dataset to a given test data point and making predictions based on the majority class (in classification) or the mean (in regression) of those K-nearest neighbors.


This implementation of KNN includes the following features:


* Customizable number of neighbors (K).
* Option to use uniform or distance-based weights for neighbor contributions.
* Support for two distance metrics: Euclidean distance (L2) and Manhattan distance (L1).
* A fit method to train the model using a training dataset.
* A predict method to make predictions on new data points.

# Parameters:

* n_neighbors (default=5): The number of neighbors to consider when making predictions.
* weights (default='uniform'): The weight function used in prediction. Options are 'uniform' (equal weights) and 'distance' (inverse of distance).
* metric (default='l2'): The distance metric used for calculating distances between data points. Options are 'l1' (Manhattan distance) and 'l2' (Euclidean distance).


# Methods:

The KNearestNeighbors class was implemented with the following attributes and methods:

n_neighbors: The number of neighbors considered for voting. weights: Determines if all neighbors have equal weight (uniform) or if closer neighbors have more influence (distance). _X: Stores the feature data upon which the model is trained. _y: Stores the target class values corresponding to _X. _distance: Holds the distance metric used; either Euclidean or Manhattan distance. The methods fit and predict were implemented as follows:

fit(X, y): Stores the training data and target values. predict(X): Predicts class target values for test data. For each test point, it calculates the distance to all training points, finds the nearest neighbors, and predicts the label based on a majority vote or a weighted vote, depending on the weights attribute.

For predict, the challenge was to efficiently calculate distances and determine the nearest neighbors. We used numpy for vectorized operations to speed up the computation.
The predict method included error handling to ensure the model was fitted before prediction. A weighted voting system was also implemented for the 'distance' weight option, where the inverse of the distance was used as a weight, enhancing the influence of closer neighbors.
