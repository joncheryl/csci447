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
    bank_sims[i] = kmn_cluster(bank, t_max=n_iterations, n_clusters=bank_n_clusters)
    wine_sims[i] = kmn_cluster(wine, t_max=n_iterations, n_clusters=wine_n_clusters)
    iris_sims[i] = kmn_cluster(iris, t_max=n_iterations, n_clusters=iris_n_clusters)
    seed_sims[i] = kmn_cluster(seed, t_max=n_iterations, n_clusters=seed_n_clusters)
    wilt_sims[i] = kmn_cluster(wilt, t_max=n_iterations, n_clusters=wilt_n_clusters)
    white_sims[i] = kmn_cluster(white, t_max=n_iterations, n_clusters=white_n_clusters)
    red_sims[i] = kmn_cluster(red, t_max=n_iterations, n_clusters=red_n_clusters)
    breast_sims[i] = kmn_cluster(breast, t_max=n_iterations, n_clusters=breast_n_clusters)
    ecoli_sims[i] = kmn_cluster(ecoli, t_max=n_iterations, n_clusters=ecoli_n_clusters)
    haber_sims[i] = kmn_cluster(haber, t_max=n_iterations, n_clusters=haber_n_clusters)

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
