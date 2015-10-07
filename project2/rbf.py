'''
Radial Basis Function Network
'''

import numpy as np
import itertools
import math

def grid(width, grain, dim):
    '''
    Build a mesh for training data.

    Example
    -------
    >>> grid(2, 3, 2)
    array([[-2., -2.],
           [-2.,  0.],
           [-2.,  2.],
           [ 0., -2.],
           [ 0.,  0.],
           [ 0.,  2.],
           [ 2., -2.],
           [ 2.,  0.],
           [ 2.,  2.]])
    '''

    oneDim = np.linspace(-width, width, grain)
    mesh = list(itertools.product(oneDim, repeat = dim))
    return np.asarray(mesh)

'''
k-means clustering function
trainSet is a 2d numpy array
nCentroids is number of centers/gaussians
wiggleRoom is convergence requirement
'''

def kmeans(trainSet, nCentroids, wiggleRoom):

    change = wiggleRoom + 1

    # initialize centers as random points in training set
    centroids = trainSet[np.random.choice(trainSet.shape[0], nCentroids, replace=False),:]
    oldCentroids = np.zeros((nCentroids, trainSet.shape[1]))

    # list to hold grouping of each point in the training set
    pointGroup = np.zeros(trainSet.shape[0])
    
    while change > wiggleRoom:
        # classify points
        for pointIndex, point in enumerate(trainSet):
            distances = np.sum((centroids - point)**2, axis=1)
            pointGroup[pointIndex] = np.argmin(distances)
        
        # find new centers
        for group in range(nCentroids):
            oldCentroids[group] = centroids[group]
            centroids[group] = np.sum(trainSet[pointGroup == group], axis=0)/sum(pointGroup == group)

        change = max(np.linalg.norm(centroids - oldCentroids, axis=1))
        
    return centroids

'''
class RBF:

    def __init__(self, nInput, nGaussians, nOutput, sigma):
        # the number of input nodes, gaussian nodes, and output nodes resp
        self.nInput = nInput
        self.nG = nGaussians
        self.nOutput = nOutput

        # not sure exactly how this is being used (normalization?)
        self.sigma = sigma

        # 2d array to hold the kmeans
        # more efficient as a list of numpy arrays?
        self.km = np.zeros((self.nInput, self.nG))


        
    # radial basis function - requires numpy arrays as input
    def rbf(x, center):
        distance = np.linalg.norm(x - center)
        return np.exp(-sigma * distance**2)

    def rbfModel(inSet, output, centers):

        #number of observations of the input dataset
        N = len(inSet)
                
        #number of variables of the input dataset
        nCol = len(inSet[0])

        # 'Unsupervised' Phase of training:
        # calculate the cluster points, mus, using kmeans
        mus = sp.cluster.kmeans(inSet, km);
                
        # calculate the Euclidean distances
        distances = (mus-mus)**2
        distances = distances.sum(axis=-1)
        distances = np.sqrt(distances)
                
        # calculate sigma
        sigma = max(distances)/sqrt(2*centers)

        # set up phi to be an empty N x K+1 matrix 
        phi = np.zeros((N,centers+1))
        for row in range(0,N):
            # set to bias column to 1
            phi[row,1] = 1
            for col in range (0,centers):
                # set the weights
                phi[row,col+1] = exp(-(1/(2*sigma))*np.linalg.norm(inSet[row:]-mus[col,])**2)
		# set the weights
		weights = np.linalg.pinv(phi.transpose()*phi) * phi.transpose() * output
            return ReturnVal(sigma, weights, mus)

        def rbfPrediction(rbfModel, X):
            sigma = rbfModel.sigma
            centers = rbfModel.centers
            weights = rbfModel.weights
            N = X.shape[0]

            prediction = np.full(N, weights[0])

            for j in range(0, N):
                for k in range (0, len(centers)):
                    prediction[j] = prediction[j] + weights[k+1]*exp(-(1/(2*sigma))*np.linalg.norm(X[j:]-centers[k,])**2)
            return prediction

class ReturnVal(object):
    def __init__(self, sigma, weights, centers):
        self.sigma = sigma
        self.weights = weights
        self.centers = centers




'''
