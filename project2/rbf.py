'''
Radial Basis Function Network
'''

import numpy as np
import itertools
import sys   # for matrix inversion check

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
    
    def __init__(self, nInput, nGaussians, nOutput):
        # the number of input nodes, gaussian nodes, and output nodes resp
        self.nInput = nInput
        self.nG = nGaussians
        self.nOutput = nOutput

        # Array of k-means determined centers of Gaussians
        self.km = np.zeros((self.nG, self.nInput))

        self.gaussVec = np.zeros(self.nG+1)
        
    def setupNetwork(self, xStart, yStart, betaDeal, wiggle):
        '''
        Setup the network so it has all the elements needed to learn via 
        Recurive Least Squares.
        1) Choose Gaussian centers
        2) Determine appropriate sigma
        3) Initialize the (xTx)^-1 like matrix and weight matrix.
        '''

        # K-means clustering
        self.km = kmeans(xStart, self.nG, wiggle)

        # variance value estimate
        oneDimDist = ((max(xStart.T[0]) - min(xStart.T[0])) / self.nG**(1./self.nInput))**2
        self.beta = self.nG**(betaDeal) / (oneDimDist * self.nInput)
        
        # Calculate the outputs from each Gaussian node for every point in
        # the initial training set
        phi = np.array([self.rbfOutputVector(x) for x in xStart])

        # Check to make sure that the resulting system has lin-ind columns
        if np.linalg.cond(np.dot(phi.T, phi)) < 1/sys.float_info.epsilon:
            self.pTpInv = np.linalg.inv(np.dot(phi.T, phi))
            self.weight = self.pTpInv.dot(phi.T).dot(yStart)            
        else:
            print 'Training data not sufficient for making initial weights :('

        # Then do it anyway
        self.pTpInv = np.linalg.inv(np.dot(phi.T, phi))
        self.weight = self.pTpInv.dot(phi.T).dot(yStart)
        
    def rbfOutputVector(self, inputVector):
        '''
        Computes output values for each Gaussian node
        a 1 is appended for the intercept term
        '''
        distance = np.sum((inputVector - self.km)**2, axis=1)
        return np.append(1., np.exp(-self.beta * distance))

    def evaluate(self, inputVector):
        '''
        Computes predicted output
        '''
        gaussOut = self.rbfOutputVector(inputVector)
        return np.dot(self.weight, gaussOut)

    def learn(self, trainingPoint, target):
        '''
        Recursive Least Squares update to weights
        '''

        # Calculate a few items that are needed for the formula
        gaussVec = self.rbfOutputVector(trainingPoint)
        denom = 1 + gaussVec.dot(self.pTpInv).dot(gaussVec)
        error = np.dot(self.weight, gaussVec) - target

        # Update weights
        self.weight -= (self.pTpInv.dot(gaussVec).dot(error)) / denom

        # Update pTp inverse matrix
        self.pTpInv -= self.pTpInv.dot(np.outer(gaussVec,gaussVec)).dot(self.pTpInv) / denom

'''
Test to see if working
'''

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

def rosen(x):
     '''
     Rosenbrock function
     '''
     x = x.T
     return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

def mse(approx, actual):
    '''
    Mean Squared Error
    '''
    return sum(np.square(actual-approx))/len(actual)

train = grid(1, 11, 2)
trainY = rosen(train)
trainBig = grid(1, 21, 2)
trainBigY = rosen(trainBig)

test = rbfNetwork(2, 100, 1)
test.setupNetwork(train, trainY, .1, .1)

resultsUntrained = np.array([test.evaluate(x) for x in trainBig])
mseUntrained = mse(resultsUntrained, trainBigY)

for i in range(len(trainBigY)): test.learn(trainBig[i],trainBigY[i])

resultsTrained =  np.array([test.evaluate(x) for x in trainBig])
mseTrained = mse(resultsTrained, trainBigY)
