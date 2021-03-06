'''
Particle Swarm Optimization clustering: gbest style

Following pseudocode from "Data Clustering using Particle Swarm Optimization"
DW van der Merwe and AP Engelbrecht

John Sherrill - CSCI 447 Fall 2015

Tunable parameters:
The w, c1, and c2 parameters were taken from the PhD thesis of F van den Bergh
2002. University of Pretoria. Referenced from der Merwe and Engelbrecht paper

t_max = 10  # number of iterations

n_particles = 10  # from der Merwe and Engelbrecht paper
n_clusters = round(data.shape[0] / 20)

w = .72  # inertial weight
c1 = 1.49  # acceleration constant 1
c2 = 1.49  # acceleration constant 2
'''

import numpy as np
import pandas as pd
from numpy import random as rd


def pso_cluster(data, t_max=1000, n_particles=10, n_clusters=3, w=.72, c1=1.49,
                c2=1.49):
    '''
    Initializations
    '''
    n_points = data.shape[0]
    n_features = data.shape[1]

    # figure out domain for particles (cluster centers)
    mins = [min(data[:, i]) for i in range(n_features)]
    maxs = [max(data[:, i]) for i in range(n_features)]
    parts = np.array([[[rd.uniform(mins[i], maxs[i])
                        for i in range(n_features)]
                       for j in range(n_clusters)]
                      for k in range(n_particles)])

    # which cluster point j is assigned to for particle i
    assigns = np.zeros((n_particles, n_points))
    # distance of point k to centroid j from particle i
    distances = np.zeros((n_particles, n_clusters, n_points))

    # fitness stuff
    fitnesses = np.zeros((n_particles))  # fitness of particle i
    # calculate distances to centroids
    for i in range(n_particles):
        for j in range(n_clusters):
            for k in range(n_points):
                distances[i, j, k] = sum((parts[i, j, :] - data[k, :])**2)

    # assign points to clusters for each particle
    assigns = np.array([np.argmin(distances[i, :, :], axis=0)
                        for i in range(n_particles)])

    # calculate quantization error (fitness of each particle)
    for i in range(n_particles):
        for j in range(n_clusters):
            temp_fit = 0
            for k in range(n_points):
                if assigns[i, k] == j:
                    temp_fit += distances[i, j, k]

            if not sum(assigns[i, :] == j) == 0:
                fitnesses[i] += temp_fit / sum(assigns[i, :] == j)

        fitnesses[i] /= n_clusters

    best_fitnesses = np.copy(fitnesses)  # local best fitnesses
    best_position = np.copy(parts)  # best position per particle

    # intermediate quantities for updating positions
    velocity = np.zeros((n_particles, n_clusters, n_features))

    '''
    Algorithm
    '''
    for t in range(t_max):

        # calculate distances to centroids
        for i in range(n_particles):
            for j in range(n_clusters):
                for k in range(n_points):
                    distances[i, j, k] = sum((parts[i, j, :] - data[k, :])**2)
                    # use sqrt just above?

        # assign points to clusters for each particle
        assigns = np.array([np.argmin(distances[i, :, :], axis=0)
                            for i in range(n_particles)])

        # calculate quantization error (fitness of each particle)
        for i in range(n_particles):
            for j in range(n_clusters):
                temp_fit = 0
                for k in range(n_points):
                    if assigns[i, k] == j:
                        temp_fit += distances[i, j, k]

                if not sum(assigns[i, :] == j) == 0:
                    fitnesses[i] += temp_fit / sum(assigns[i, :] == j)

            fitnesses[i] /= n_clusters

        # update local best positions (personal best for each particle)
        for i in range(n_particles):
            if fitnesses[i] < best_fitnesses[i]:
                best_position[i] = np.copy(parts[i, :, :])
                best_fitnesses[i] = np.copy(fitnesses[i])

        # update global best position (best of personal bests)
        global_best = np.copy(best_position[np.argmin(best_fitnesses), :, :])

        # update cluster centroids for each particle
        r1 = np.random.uniform(0, 1, n_clusters)
        r2 = np.random.uniform(0, 1, n_clusters)
        for i in range(n_particles):
            for k in range(n_features):
                velocity[i, :, k] = w * velocity[i, :, k] + \
                    c1 * r1[:] * (best_position[i, :, k] - parts[i, :, k]) + \
                    c2 * r2[:] * (global_best[:, k] - parts[i, :, k])

            parts[i, :, :] += velocity[i, :]

        #print(np.min(best_fitnesses))
        return(np.min(best_fitnesses))


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

wz = 0.72
c1z = 1.49
c2z = 1.49
n_simulations = 30
n_iterations = 10

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
    bank_sims[i] = pso_cluster(bank, t_max=n_iterations, n_clusters=bank_n_clusters, w=wz, c1=c1z, c2=c2z)
    wine_sims[i] = pso_cluster(wine, t_max=n_iterations, n_clusters=wine_n_clusters, w=wz, c1=c1z, c2=c2z)
    iris_sims[i] = pso_cluster(iris, t_max=n_iterations, n_clusters=iris_n_clusters, w=wz, c1=c1z, c2=c2z)
    seed_sims[i] = pso_cluster(seed, t_max=n_iterations, n_clusters=seed_n_clusters, w=wz, c1=c1z, c2=c2z)
    wilt_sims[i] = pso_cluster(wilt, t_max=n_iterations, n_clusters=wilt_n_clusters, w=wz, c1=c1z, c2=c2z)
    white_sims[i] = pso_cluster(white, t_max=n_iterations, n_clusters=white_n_clusters, w=wz, c1=c1z, c2=c2z)
    red_sims[i] = pso_cluster(red, t_max=n_iterations, n_clusters=red_n_clusters, w=wz, c1=c1z, c2=c2z)
    breast_sims[i] = pso_cluster(breast, t_max=n_iterations, n_clusters=breast_n_clusters, w=wz, c1=c1z, c2=c2z)
    ecoli_sims[i] = pso_cluster(ecoli, t_max=n_iterations, n_clusters=ecoli_n_clusters, w=wz, c1=c1z, c2=c2z)
    haber_sims[i] = pso_cluster(haber, t_max=n_iterations, n_clusters=haber_n_clusters, w=wz, c1=c1z, c2=c2z)

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
