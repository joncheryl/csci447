'''
K-Means clustering

Parameters
----------
data : 2-D ndarray
    training set input values
n_clusters : number
    number of centers/gaussians
wiggleRoom : number
    convergence requirement
'''

import pandas as pd
import numpy as np
from numpy import random as rd
from numpy import linalg as la


def kmn_cluster(data, t_max=1000, n_clusters=3):

    n_points = data.shape[0]

    # initialize centers as random points in training set
    centroids = data[rd.choice(data.shape[0], n_clusters, replace=False), :]
    oldCentroids = np.zeros((n_clusters, data.shape[1]))

    # list to hold grouping of each point in the training set
    pointGroup = np.zeros(n_points)

    for t in range(t_max):
        # classify points
        for pointIndex, point in enumerate(data):
            distances = np.sum((centroids - point)**2, axis=1)
            pointGroup[pointIndex] = np.argmin(distances)

        # find new centers
        for group in range(n_clusters):
            oldCentroids[group] = centroids[group]
            centroids[group] = np.sum(data[pointGroup == group], axis=0) / \
                sum(pointGroup == group)

        # calculate quantization error
        quant_error = 0
        for i in range(n_clusters):
            # average distance to center in each cluster
            temp_fit = 0
            for k in range(n_points):
                if pointGroup[k] == i:
                    temp_fit += la.norm(data[k] - centroids[i])
            quant_error += temp_fit / sum(pointGroup == i)
        # average of averages
        quant_error /= n_clusters
        print(quant_error)


# Read in data
bank_data = pd.read_csv("data/banknote.csv", header=None).as_matrix()
wine_data = pd.read_csv("data/wine_reordered.csv", header=None).as_matrix()
iris_data = pd.read_csv("data/iris.csv", header=None).as_matrix()
seed_data = pd.read_csv("data/seeds_dataset.csv", header=None).as_matrix()
wilt_data = pd.read_csv("data/wilt_training_reordered.csv",
                        header=None).as_matrix()

# Number of clusters is the number of categories
bank_n_clusters = np.unique(bank_data[:, -1]).size
wine_n_clusters = np.unique(wine_data[:, -1]).size
iris_n_clusters = np.unique(iris_data[:, -1]).size
seed_n_clusters = np.unique(seed_data[:, -1]).size
wilt_n_clusters = np.unique(wilt_data[:, -1]).size

# Get rid of categories
bank = bank_data[:, 0:-1]
wine = wine_data[:, 0:-1]
iris = iris_data[:, 0:-1]
seed = seed_data[:, 0:-1]
wilt = wilt_data[:, 0:-1]

n_simulations = 1
n_iterations = 100

for i in range(n_simulations):
    kmn_cluster(bank, t_max=n_iterations, n_clusters=bank_n_clusters)
    kmn_cluster(wine, t_max=n_iterations, n_clusters=wine_n_clusters)
    kmn_cluster(iris, t_max=n_iterations, n_clusters=iris_n_clusters)
    kmn_cluster(seed, t_max=n_iterations, n_clusters=seed_n_clusters)
    kmn_cluster(wilt, t_max=n_iterations, n_clusters=wilt_n_clusters)
