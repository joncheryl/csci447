'''
Ant Colony Optimization clustering: Implimentation of Fuzzy c-Means Model

Following pseudocode from "Ant Colony Optimization of Clustering Models" by
Thomas Runkler

John Sherrill - CSCI 447 Fall 2015

data = pd.read_csv("data/seeds_dataset.txt", sep='\t').as_matrix()

Tunable parameters:

t_max = 100  # number of iterations
n_clusters = round(data.shape[0] / 20)
epsilon = 0.1  # what is this? Must be non-negative
rho = 0.5  # what is this? Must be in [0,1]
alpha = 1.5  # what is this? Must be >= 1
'''

import numpy as np
from numpy import linalg as la
import pandas as pd


def aco_cluster(data, t_max=1000, n_particles=10, n_clusters=3, epsilon=0.1,
                rho=0.5, alpha=2):

    '''
    Initializations
    '''
    n_points = data.shape[0]
    pheromones = np.ones((n_clusters, n_points))
    j_hcm_min = np.inf  # objective function

    for t in range(t_max):

        # initialize cluster selections
        u_clusters = np.zeros((n_clusters, n_points))

        # randomly generate clusters until all contain at least one point
        while(not np.all([np.any(u_clusters[i, :])
                          for i in range(n_clusters)])):

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

        # calculate quantization error
        quant_error = 0
        for i in range(n_clusters):
            # average distance to center in each cluster
            temp_fit = 0
            for k in range(n_points):
                if u_clusters[i, k] == 1:
                    temp_fit += la.norm(data[k] - v_centers[i])
            quant_error += temp_fit / sum(u_clusters[i, :])
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
n_iterations = 1000
rhoz = 0.005  # from paper (page 1242, bottom)
epz = 0.3  # from paper (page 1242, bottom)  .3 works best for wine
alphaz = 1  # from paper (page 1242, bottom)

for i in range(n_simulations):
    aco_cluster(bank, t_max=n_iterations, n_clusters=bank_n_clusters,
                epsilon=epz, rho=rhoz, alpha=alphaz)
    aco_cluster(wine, t_max=n_iterations, n_clusters=wine_n_clusters,
                epsilon=epz, rho=rhoz, alpha=alphaz)
    aco_cluster(iris, t_max=n_iterations, n_clusters=iris_n_clusters,
                epsilon=epz, rho=rhoz, alpha=alphaz)
    aco_cluster(seed, t_max=n_iterations, n_clusters=seed_n_clusters,
                epsilon=epz, rho=rhoz, alpha=alphaz)
    aco_cluster(wilt, t_max=n_iterations, n_clusters=wilt_n_clusters,
                epsilon=epz, rho=rhoz, alpha=alphaz)
