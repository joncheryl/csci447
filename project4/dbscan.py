#!/usr/bin/python
#
# This is an implementation of DBScan in which points are categorized as core,
# boundary, or noise initially and then assigned to clusters.  Epsilon values
# and MinPts are automatically determined using heuristics, and the 
# neighborhood function uses Euclidean distances.
#

import numpy as np
import pandas as pd
import scipy.special as sp


def distance(vec1, vec2):
    return np.sum(np.abs(vec1-vec2))/len(vec1)

def normalizeRows(M):
    return M / M.sum(axis=1)[:,np.newaxis]    

def quant_err(clustered, dat):
    maxClusters = np.max(clustered).astype(int) + 1
    C = np.zeros(maxClusters)    
    Nc = np.unique(clustered).shape[0]
    
    centroids = findCentroids(clustered, dat)
    numerator = 0
    
    for j in range(maxClusters):
    # skip non-clustered points
        if j==0:
            continue
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

def categorize(D, eps, minPts):
    nPoints = D.shape[0]
    categorized = np.zeros(nPoints)
    
    # Find cores and border points.  There's probably a much more efficient 
    # way to do all of this.
    
    for i in range(nPoints):
        regionCount = 0
        for j in range(nPoints):
            if j == i:
                continue
            if distance(D[i], D[j]) < eps:
                regionCount = regionCount + 1
        # Core points
        if regionCount >= minPts:
            categorized[i] = 1
        # Border points
        if regionCount > 0 and regionCount < minPts:
            categorized[i] = 2
    return categorized    
    
    
def dbscan(D, core, nonNoise, eps, minPts):
    clustLbl = np.zeros(D.shape[0])
    currClustLbl = 0
    for p in core:
        if clustLbl[p] == 0:
            currClustLbl = currClustLbl + 1
            clustLbl[p] = currClustLbl
        for pPrime in neighborhood(D[p], D, nonNoise, eps):
            if clustLbl[pPrime] == 0:
                clustLbl[pPrime] = currClustLbl
    return clustLbl
    
def neighborhood(point, dat, pointSet, eps):
    nbd = []
    for candidate in pointSet:
        if np.all(point == dat[candidate]):
            continue
        if distance(point, dat[candidate]) < eps:
            nbd.append(candidate)
    return nbd


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

nRuns=1
runs = np.zeros(shape=(files.shape[0],nRuns))
clusters = np.zeros(shape=(files.shape[0],nRuns))

print "Starting DBScan:"
for dSet in range(files.shape[0]):
    data = pd.read_csv(files[dSet][1], header=None).as_matrix()
    nClust = np.unique(data[:,-1]).shape[0]
    data = data[:, 0:-1]

    n_pts, n_att =data.shape
    maxVal = data.max(axis=0)
    minVal = data.min(axis=0)

    nClust =  np.unique(data[:,-1]).size
    minPts = max(np.ceil(n_pts/(nClust)),2)
    
    num = np.prod(maxVal-minVal)*nClust*sp.gamma(.5*n_att+1)
    den = n_pts*np.sqrt(np.pi**n_att)
    eps = ((np.prod(maxVal-minVal)*minPts*sp.gamma(.5*n_att+1))/ \
          (n_pts*np.sqrt(np.pi**n_att)))**(1.0/n_att)  
    categorized = categorize(data, eps, minPts)
    core = np.where(categorized == 1)[0]
    
    # set of core and border points for calculating neighborhoods
    borderCore = np.concatenate((np.where(categorized == 1), \
                 np.where(categorized == 2)), axis=1)[0]
    borderCore.sort() 

    clustered = dbscan(data,core,borderCore,eps, minPts)
    nClusters = np.max(clustered)
    print "--------------------------"
    print "--------------------------"
    print files[dSet][0], "Data Set"
    print "--------------------------"
    for run in range(nRuns):
        print "Run %d:" %(run+1) 
        runs[dSet,run] = quant_err(clustered, data)
        print "Quantization Error: ", runs[dSet,run]
        clusters[dSet,run] = np.max(clustered).astype(int)
        print clusters[dSet,run], "clusters"
    print "Epsilon = %.4f, minPts = %d" % (eps, minPts)        
    print "--------------------------"
    print "--------------------------"
    print    
    
print "Quantitative Error:"
print "Data Set\t\t Mean\t SD"
for result in range(runs.shape[0]):
    print "%s\t\t%.3f\t%.3f" % (files[result][0], np.mean(runs[result]), np.std(runs[result]) )
print         
print "Clusters:"
print "Data Set\t\tClusters"
for result in range(runs.shape[0]):
    print "%s\t\t%.3f" % (files[result][0], clusters[result])

pd.DataFrame(runs).to_csv("dbscan_runs.csv", header=False, index=False)