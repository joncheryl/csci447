'''
Ant Colony Optimization clustering: Implimentation of Fuzzy c-Means Model

Following pseudocode from "Ant Colony Optimization of Clustering Models" by
Thomas Runkler

John Sherrill - CSCI 447 Fall 2015
'''

import numpy as np
from numpy import linalg as la
import pandas as pd


def aco_cluster(data, t_max=1000, n_clusters=3, epsilon=0.1, rho=0.5, alpha=2):

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

    return(quant_error)

# Read in data
bank_data = pd.read_csv("data/banknote.csv", header=None).as_matrix()
wine_data = pd.read_csv("data/wine_reordered.csv", header=None).as_matrix()
iris_data = pd.read_csv("data/iris.csv", header=None).as_matrix()
seed_data = pd.read_csv("data/seeds_dataset.csv", header=None).as_matrix()
wilt_data = pd.read_csv("data/wilt_training_reordered.csv",
                        header=None).as_matrix()
white_data = pd.read_csv("data/winequality-white.csv", header=None).as_matrix()
red_data = pd.read_csv("data/winequality-red.csv", header=None).as_matrix()
breast_data = pd.read_csv("data/BreastTissue.csv", header=None).as_matrix()
ecoli_data = pd.read_csv("data/ecoli.csv", header=None).as_matrix()
haber_data = pd.read_csv("data/haberman.csv", header=None).as_matrix()

# Number of clusters is the number of categories
bank_n_clusters = np.unique(bank_data[:, -1]).size
wine_n_clusters = np.unique(wine_data[:, -1]).size
iris_n_clusters = np.unique(iris_data[:, -1]).size
seed_n_clusters = np.unique(seed_data[:, -1]).size
wilt_n_clusters = np.unique(wilt_data[:, -1]).size
white_n_clusters = np.unique(white_data[:, -1]).size
red_n_clusters = np.unique(red_data[:, -1]).size
breast_n_clusters = np.unique(breast_data[:, -1]).size
ecoli_n_clusters = np.unique(ecoli_data[:, -1]).size
haber_n_clusters = np.unique(haber_data[:, -1]).size

# Get rid of categories
bank = bank_data[:, 0:-1]
wine = wine_data[:, 0:-1]
iris = iris_data[:, 0:-1]
seed = seed_data[:, 0:-1]
wilt = wilt_data[:, 0:-1]
white = white_data[:, 0:-1]
red = red_data[:, 0:-1]
breast = breast_data[:, 0:-1]
ecoli = ecoli_data[:, 0:-1]
haber = haber_data[:, 0:-1]

rhoz = .1  # from paper (page 1242, bottom)  .005
epz = .01  # from paper (page 1242, bottom)  .3 works best for wine
alphaz = 1  # from paper (page 1242, bottom)  1
n_simulations = 30
n_iterations = 3

bank_sims = np.zeros(n_simulations)
wine_sims = np.zeros(n_simulations)
iris_sims = np.zeros(n_simulations)
seed_sims = np.zeros(n_simulations)
wilt_sims = np.zeros(n_simulations)
white_sims = np.zeros(n_simulations)
red_sims = np.zeros(n_simulations)
breast_sims = np.zeros(n_simulations)
ecoli_sims = np.zeros(n_simulations)
haber_sims = np.zeros(n_simulations)

for i in range(n_simulations):
    bank_sims[i] = aco_cluster(bank, t_max=n_iterations, n_clusters=bank_n_clusters, rho=.1, epsilon=.1, alpha=1)
    wine_sims[i] = aco_cluster(wine, t_max=n_iterations, n_clusters=wine_n_clusters, rho=.005, epsilon=.3, alpha=1)
    iris_sims[i] = aco_cluster(iris, t_max=n_iterations, n_clusters=iris_n_clusters, rho=.1, epsilon=.5, alpha=.5)
    seed_sims[i] = aco_cluster(seed, t_max=n_iterations, n_clusters=seed_n_clusters, rho=.1, epsilon=.8, alpha=.8)
    wilt_sims[i] = aco_cluster(wilt, t_max=n_iterations, n_clusters=wilt_n_clusters, rho=.1, epsilon=.1, alpha=.8)
    red_sims[i] = aco_cluster(red, t_max=n_iterations, n_clusters=red_n_clusters, rho=.001, epsilon=.2, alpha=.5)
    white_sims[i] = aco_cluster(white, t_max=n_iterations, n_clusters=white_n_clusters, rho=.001, epsilon=.5, alpha=.5)
    breast_sims[i] = aco_cluster(breast, t_max=n_iterations, n_clusters=breast_n_clusters, rho=.00001, epsilon=.005, alpha=.5)
    ecoli_sims[i] = aco_cluster(ecoli, t_max=n_iterations, n_clusters=ecoli_n_clusters, rho=.01, epsilon=.1, alpha=1)
    haber_sims[i] = aco_cluster(haber, t_max=n_iterations, n_clusters=haber_n_clusters, rho=.1, epsilon=.01, alpha=1)

print("bank mean:", np.mean(bank_sims))
print("wine mean:", np.mean(wine_sims))
print("iris mean:", np.mean(iris_sims))
print("seed mean:", np.mean(seed_sims))
print("wilt mean:", np.mean(wilt_sims))
print("white mean:", np.mean(white_sims))
print("red mean:", np.mean(red_sims))
print("breast mean:", np.mean(breast_sims))
print("ecoli mean:", np.mean(ecoli_sims))
print("haber mean:", np.mean(haber_sims))

print("bank sd", np.std(bank_sims))
print("wine sd", np.std(wine_sims))
print("iris sd", np.std(iris_sims))
print("seed sd", np.std(seed_sims))
print("wilt sd", np.std(wilt_sims))
print("white sd", np.std(white_sims))
print("red sd", np.std(red_sims))
print("breast sd", np.std(breast_sims))
print("ecoli sd", np.std(ecoli_sims))
print("haber sd", np.std(haber_sims))
