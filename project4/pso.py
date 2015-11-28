'''
Particle Swarm Optimization clustering: gbest style

Following pseudocode from "Data Clustering using Particle Swarm Optimization"
DW van der Merwe and AP Engelbrecht

John Sherrill - CSCI 447 Fall 2015
'''

import numpy as np
from numpy import random as rd
from numpy import linalg as la

import pandas as pd
data = pd.read_csv("data/seeds_dataset.txt", sep='\t').as_matrix()
n_points = data.shape[0]
n_features = data.shape[1]

# figure out domain for particles (cluster centers)
mins = [min(data[:, i]) for i in range(n_features)]
maxs = [max(data[:, i]) for i in range(n_features)]

# tunable parameters
n_particles = 10
n_clusters = round(data.shape[0] / 20)
w = 1  # inertial weight
c1 = 1  # acceleration constant 1
c2 = 1  # acceleration constant 2

t_max = 100  # number of iterations

# initialize swarm, assignments, distances to nearest centroid, fitnesses
parts = np.array([[[rd.uniform(mins[i], maxs[i]) for i in range(n_features)]
                   for j in range(n_clusters)] for k in range(n_particles)])
assigns = np.zeros((n_points, n_particles))
distances = np.zeros((n_particles, n_clusters, n_points))
fitnesses = np.zeros((n_particles))
velocity = np.zeros((n_particles, n_clusters))

best_fitnesses = np.zeros((n_particles))  # CHANGE THIS!
best_position = parts

for t in range(t_max):
    print(t)
    # calculate distances to centroids
    for i in range(n_particles):
        for j in range(n_clusters):
            for k in range(n_points):
                distances[i, j, k] = la.norm(parts[i, j, :] - data[k, :])**2

    # make assignments
    assigns = np.array([np.argmin(distances[i, :, :], axis=0)
                        for i in range(n_particles)])

    # calculate quantization error
    for i in range(n_particles):
        for j in range(n_clusters):
            temp_fit = 0
            for k in range(n_points):
                if assigns[i, k] == j:
                    temp_fit += distances[i, j, k]

            if not sum(assigns[i, :] == j) == 0:
                fitnesses[i] += temp_fit / sum(assigns[i, :] == j)

        fitnesses[i] /= n_clusters

    # get local best positions
    for i in range(n_particles):
        if fitnesses[i] < best_fitnesses[i]:
            best_position[i] = parts[i, :, :]

    # and global best
    '''
    this totally doesn't work right now
    '''
    global_best = argmax(fitness(best_position[i]) for i in n_particles)

    # update cluster centroids for each particle
    '''
    I don't know the proper way to use these random components based on the
    notation from the paper, which is not super well written
    '''
    r1 = np.random.uniform(0, 1, n_clusters)
    r2 = np.random.uniform(0, 1, n_clusters)

    for i in range(n_particles):
        for j in range(n_clusters):
            velocity[i, j] = w * velocity[i, j] + \
                c1 * r1[j] * (best_position[i, j, :] - parts[i, j, :]) + \
                c2 * r2[j] * (global_best[j, :] - parts[i, j, :])

        parts[i, :, :] += velocity[i, :]
