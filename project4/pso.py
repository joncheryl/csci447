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

    print(np.min(best_fitnesses))
