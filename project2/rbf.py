'''
Radial Basis Function Network
'''

import numpy as np
import itertools
import sys   # for matrix inversion check
# do we need this?
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

def kmeans(trainSet, nCentroids, wiggleRoom):
    '''
    K-Means clustering function

    Parameters
    ----------
    trainSet : 2-D ndarray
        training set input values
    nCentroids : number
        number of centers/gaussians
    wiggleRoom : number
        convergence requirement
    '''

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

class rbfNetwork:
    '''
    Radial Basis Function Network
    '''
    
    def __init__(self, nInput, nGaussians, nOutput, sigma, xStart, yStart):
        # the number of input nodes, gaussian nodes, and output nodes resp
        self.nInput = nInput
        self.nG = nGaussians
        self.nOutput = nOutput

        # normalization constant
        self.sigma = sigma

        # Array of k-means determined centers of Gaussians
        self.km = np.zeros((self.nG, self.nInput))

        # Matrix to hold the 'XtX matrix' for learning
        self.xtx = np.transpose(xStart).dot(xStart)

        if np.linalg.cond(self.xtx) < 1/sys.float_info.epsilon:
            self.weight = np.linalg.inv(self.xtx).dot(np.transpose(xStart)).dot(yStart)
        else:
            print 'initial training data not of full rank'
        
    def setupNetwork(self, trainSet, wiggle):
        '''
        Choose Gaussian centers.
        Should/Could do this with xStart data.
        Maybe this trainSet would be too big.
        '''
        self.km = kmeans(trainSet, self.nG, wiggle)
        
    def evaluate(self, inputVector):
        distance = np.sum((inputVector - self.km)**2, axis=1)
        gaussOut = np.exp(distance / -self.sigma)

        return np.dot(np.transpose(gaussOut), self.weight)

    def learn(self, trainingPoint, target):
        '''
        Can easily change to let target value be last dimension in
        trainingPoint
        '''

#        self.pMatrix += 
