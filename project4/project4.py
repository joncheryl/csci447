'''
Master script for Project 4

Brandon Fenton
John Sherrill
'''

import pandas as pd
import numpy as np
from pso import pso_cluster

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
n_iterations = 10
for i in range(n_simulations):
    pso_cluster(bank, t_max=n_iterations, n_clusters=bank_n_clusters)
    pso_cluster(wine, t_max=n_iterations, n_clusters=wine_n_clusters)
    pso_cluster(iris, t_max=n_iterations, n_clusters=iris_n_clusters)
    pso_cluster(seed, t_max=n_iterations, n_clusters=seed_n_clusters)
    pso_cluster(wilt, t_max=n_iterations, n_clusters=wilt_n_clusters)
