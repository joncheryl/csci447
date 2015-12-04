#!/usr/bin/python
#
# This is n implementation of competitive learning using similarity as 
# defined by the inverse Euclidean distance to determine fitness, and node  
# biases to prevent dead neurons.
#
# 1. Normalize all input patterns
# 2.   Randomly select a pattern x_n
# 2.a. Find find the winner neuron 
# 2.b. Update the winner neuron w_i = w_i + eta*x_n
# 2.c. Normalize the winner neuron w_i = w_i/||w_i||
# 3. Go to step 2 until no changes occur in Nex runs
# 
import numpy as np
import pandas as pd

def normalizeRows(M):
    return M / M.sum(axis=1)[:,np.newaxis]

def normalize(V):
    nV =  V / np.sum(V)
    return nV

def similarity(vec1, vec2):
    return 1/(1 + np.sum(np.abs(vec1-vec2)))
    
def distance(vec1, vec2):
    return np.sum(np.abs(vec1-vec2))

def quant_err(clustered, dat):
    # Implements a quantization error function
    maxClusters = np.max(clustered).astype(int) + 1
    C = np.zeros(maxClusters)    
    Nc = np.unique(clustered).shape[0]
    
    centroids = findCentroids(clustered, dat)
    numerator = 0
    
    for j in range(maxClusters):
        if np.argwhere(clustered == j).shape[0] == 0:
            continue
        
        C[j] = np.argwhere(clustered == j).shape[0]
        
        for k in np.where(clustered==j)[0]:
            numerator = numerator + \
                np.sum(np.abs(dat[k] - centroids[j])) / C[j]
    qErr = numerator/Nc
    return qErr
    
def findCentroids(clustered, dat):  
    maxClusters = np.max(clustered).astype(int) + 1
    
    centroids = np.zeros(shape = (maxClusters, dat.shape[1]))
    
    for clust in range(maxClusters):
        if np.argwhere(clustered == clust).shape[0] == 0:
            continue
        if np.argwhere(clustered == clust).shape[0] == 1:
            centroids[clust] = dat[np.where(clustered==clust),]
            continue
        
        centroids[clust] = \
        dat[np.where(clustered==clust),].sum(axis = 1) / \
        np.where(clustered == clust)[0].shape[0]
        
    return centroids




def competitive_learning(data, nclust=5):
    normalized = normalizeRows(data)

    # Tunable parameters
    # 
    eta = 0.25
    
    # Called it maxClusters instead of nClusters because there
    # are sometimes "dead neurons."  The biases help prevent this,
    # but it still happens
    maxClust = nclust
    Nex = 10
    nChange = 0
    tol = 0.0005
    
    # initialize weights
    w = normalizeRows(np.random.uniform(size=(maxClust, normalized.shape[1])))
    biases = np.zeros(maxClust)

    # learn
    iterations = 1
    while nChange < Nex:
        output  = np.zeros(maxClust)
        
        x_n = normalized[np.random.choice(normalized.shape[0],size=1)[0]]
        
        sims = np.zeros(w.shape[0])
    
        for j in range(w.shape[0]):
            sims[j] = similarity(w[j],x_n) + biases[j]/(iterations)
        
        winner = np.argmin(sims)
        output[winner] = 1
        biases[winner] = biases[winner] + 1
        w[winner] = (w[winner] + eta * x_n) / np.sum(w[winner] + eta * x_n)
        iterations = iterations + 1
        if (np.max(biases/iterations) - np.min(biases/iterations) < tol)  == True:
            nChange = nChange + 1
        else:
            nChange = 0
    
    # cluster
    clusters = np.zeros(normalized.shape[0])
    for i in range(normalized.shape[0]):
        sims = np.zeros(w.shape[0])
            
        for j in range(w.shape[0]):
            sims[j] = similarity(w[j],normalized[i]) + biases[j]/iterations
        
        winner = np.argmin(sims)
        clusters[i] = winner

    return clusters, iterations
 


files = np.array([["Bank Note","data/banknote.csv"],
         ["Wine", "data/wine_reordered.csv"],
         ["Iris", "data/iris.csv"],
         ["Seeds", "data/seeds_dataset.csv"],
         ["Wilt", "data/wilt_training_reordered.csv"],
         ["White", "data/winequality-white.csv"],
         ["Red", "data/winequality-red.csv"],
         ["Breast", "data/BreastTissue.csv"],
         ["E. Coli", "data/ecoli.csv"],
         ["Haberman", "data/haberman.csv"]])

nRuns=30
runs = np.zeros(shape=(files.shape[0],nRuns))
convergence = np.zeros(shape=(files.shape[0],nRuns))
neurons = np.zeros(shape=(files.shape[0],nRuns))

print "Starting Competitive Learning:"
for dSet in range(files.shape[0]):
    data = pd.read_csv(files[dSet][1], header=None).as_matrix()
    n_clust = np.unique(data[:,-1]).shape[0]
    data = data[:, 0:-1]
    normalized = normalizeRows(data)
    print "--------------------------"
    print "--------------------------"
    print files[dSet][0], "Data Set"
    print "--------------------------"
    for run in range(nRuns):
        print "Run %d:" %(run+1)
        clustrd, converged = competitive_learning(normalized, n_clust) 
        runs[dSet,run] = quant_err(clustrd, data)
        convergence[dSet,run] = converged
        print "Quantization Error: ", runs[dSet,run]
        neurons[dSet,run] = np.max(clustrd).astype(int) + 1
        print neurons[dSet,run], "active neurons,", \
              n_clust-neurons[dSet,run], "dead neurons"
    print "--------------------------"
    print "--------------------------"
    print    
    
print "Quantization Error:"
print "Data Set\t\tMean\tSD"
for result in range(runs.shape[0]):
    print "%s\t\t%.3f\t%.3f" % (files[result][0], np.mean(runs[result]), np.std(runs[result]) )
print         
print "Convergence:"
print "Data Set\t\tMean\tSD"
for result in range(runs.shape[0]):
    print "%s\t\t%.3f\t%.3f" % (files[result][0], np.mean(convergence[result]), np.std(convergence[result]) )
print
print "Clusters:"
print "Data Set\t\tMean\tSD"
for result in range(runs.shape[0]):
    print "%s\t\t%.3f\t%.3f" % (files[result][0], np.mean(neurons[result]), np.std(neurons[result]) )

pd.DataFrame(runs).to_csv("cl_runs.csv", header=False, index=False)