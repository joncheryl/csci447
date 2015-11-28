'''
Ant Colony Optimization clustering: Implimentation of Fuzzy c-Means Model

Following pseudocode from "Ant Colony Optimization of Clustering Models" by
Thomas Runkler

John Sherrill - CSCI 447 Fall 2015
'''

import numpy as np
from numpy import linalg as la
import pandas as pd
data = pd.read_csv("data/seeds_dataset.txt", sep='\t').as_matrix()
n_points = data.shape[0]

# tunable parameters
n_clusters = round(data.shape[0] / 20)
t_max = 100  # number of iterations
epsilon = 0.1  # what is this? Must be non-negative
rho = 0.5  # what is this? Must be in [0,1]
alpha = 1.5  # what is this? Must be >= 1

# initialization
pheromones = np.ones((n_clusters, n_points))
j_hcm_min = np.inf  # objective function

for t in range(t_max):

    # initialize cluster selections
    u_clusters = np.zeros((n_clusters, n_points))

    # randomly generate clusters until all clusters contain at least one point
    while(not np.all([np.any(u_clusters[i, :]) for i in range(n_clusters)])):
        # initialize cluster selections
        u_clusters = np.zeros((n_clusters, n_points))

        # probabilities a particular cluster choice is made
        probs = np.array([pheromones[:, k] / sum(pheromones[:, k])
                          for k in range(n_points)])
        for k in range(n_points):
            u_clusters[np.random.choice(n_clusters, p=probs[k]), k] = 1

    # calculate cluster centers (average points in clusters)
    v_centers = [np.dot(data.T, u_clusters[i, :]) /
                 sum(u_clusters[i, :])
                 for i in range(n_clusters)]

    # compute objective function and potentially update
    j_hcm = sum(sum(u_clusters[i, k] * la.norm(v_centers[i] - data[k])**2
                    for k in range(n_points)) for i in range(n_clusters))
    if j_hcm_min > j_hcm:
        j_hcm_min = j_hcm

    # update pheromones
    for i in range(n_clusters):
        for k in range(n_points):
            pheromones[i, k] = pheromones[i, k] * (1 - rho) + \
                u_clusters[i, k] / (j_hcm - j_hcm_min + epsilon)**alpha
