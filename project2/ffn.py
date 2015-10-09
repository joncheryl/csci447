'''
still need to get biases to learn
'''

import numpy as np
import itertools
import math

from operator import sub # for testing

def sigmoid(x):
     if x>0:
         return 1 / (1 + math.exp(-x))
     else:
         return math.exp(x) / (math.exp(x) + 1)

def dsigmoid(x):
    return math.exp(-x)/(1+math.exp(-x))**2
    
class ffNetwork:

    def __init__(self, sizes, learningRate):
        '''
        Each element in sizes contains the number of nodes in each
        respective layer. For example, if sizes is of length 3, sizes[0] would
        be how many input nodes are specified, sizes[1] is how many nodes are
        in the single hidden layer and sizes[2] is the number of output nodes
        '''
        self.sizes = sizes
        self.a = learningRate
        
        # Build network topology
        self.nodes = [[inputNode(i) for i in range(sizes[0])]]
        self.edges = []

        for layer in range(len(sizes))[1:-1]:
            self.nodes.append( [Node() for j in range(sizes[layer])] )
            for j in range(sizes[layer]):
                for k in range(sizes[layer-1]):
                    self.connect(self.nodes[layer-1][k], self.nodes[layer][j], np.random.randn(1))

        self.nodes.append([outputNode(i) for i in range(sizes[-1])])
        for j in range(sizes[-1]):
            for k in range(sizes[-2]):
                self.connect(self.nodes[-2][k], self.nodes[-1][j], np.random.randn(1))
        
    def connect(self, upstream, downstream, weight):
        connector = Edge(upstream, downstream, weight, self.a)
        self.edges += [connector]
        upstream.addOutput(connector)
        downstream.addInput(connector)
                    
    # input must be a vector of length sizes[0]
    # could be a numpy (n,1) array?
    def feedforward(self, inputVector):
        self.output = [i.feedforward(inputVector) for i in self.nodes[-1]]
        return self.output

    def learn(self, inputVector, target):
        # Compute outputs for each node
        self.feedforward(inputVector)

        # Backpropogate errors
        for inputNode in self.nodes[0]:
            for edge in inputNode.outputs:
                edge.down.backProp(target)

        # Update weights
        for edge in self.edges:
            # w = w -learningRate * dEdOutput * dOutputdInput * dInputdWeight
            edge.weight -= edge.a * edge.down.delta * edge.up.output

    def getWeights(self):
         weights=[]
         for edge in self.edges:
              weights.append(edge.weight[0])
         return(weights)
        
class Edge:
    def __init__(self, upstream, downstream, weight, learningRate):
        # upstream and downstream nodes
        self.up = upstream
        self.down = downstream

        self.weight = weight

        '''
        having the learning rate defined here may potentially allow for
        different learning rates at different levels of the network. May be
        interesting to test, time permitting
        '''
        self.a = learningRate

    def getOutput(self, inputVector):
        return self.up.feedforward(inputVector) * self.weight
                           
class Node:

    def __init__(self):
        self.netInput = 0
        self.output = 0
        # weights may need to be stored in upstream node for learning
        self.inputs = []
        self.outputs = []
        self.bias = np.random.randn(1)

        self.delta = 0
        self.expectedOutput = 0
        
    def addInput(self, edge):
        self.inputs.append(edge)

    def addOutput(self, edge):
        self.outputs.append(edge)
        
    def feedforward(self, inputVector):
        '''
        adding zero makes it work... I have no idea why. 
        It's like self.bias calls np.random.randn(1) everytime or something
        '''
        
        self.netInput = 0 # + self.bias
        for i in self.inputs:
            self.netInput += i.getOutput(inputVector)

        self.output = sigmoid(self.netInput)
        return self.output

    def backProp(self, target):
        self.delta = 0

        # dEdOutput
        for outEdge in self.outputs:
            self.delta += outEdge.down.backProp(target) * outEdge.weight

        # dOutputdInput
        self.delta *= dsigmoid(self.netInput)

        return self.delta

# does this need to inheret from the Node class?
class inputNode(Node):

    def __init__(self, index):
        self.index = index
        self.output = 0
        self.outputs = []

    def addOutput(self, edge):
        self.outputs.append(edge)

    def feedforward(self, inputVector):
        self.output = inputVector[self.index]
        return self.output

class outputNode(Node):

    def __init__(self, index):
        self.index = index
        self.netInput = 0
        self.inputs = []
        self.output = 0
        self.bias = np.random.randn(1)

        self.delta = 0
        
    def addInput(self, edge):
        self.inputs.append(edge)

    def feedforward(self, inputVector):
        self.netInput = 0 # + self.bias
        for i in self.inputs:
            self.netInput += i.getOutput(inputVector)

        self.output = self.netInput
        return self.output[0]

    def backProp(self, target):
        # dEdOutput
        self.delta = (self.output - target[self.index])

        # dOutputdInput = 1 so nothing to multiply
        return self.delta


'''
Test network
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

'''
train = grid(1, 11, 2)
trainY = rosen(train)
trainBig = grid(1, 21, 2)
trainBigY = rosen(trainBig)

topology = np.array([2, 4, 4, 1])
learningRate = .5
nIter = 10
tol = .0001

converged = 0

testFF = ffNetwork(topology, learningRate)
for iteration in range(nIter):
    oldWeights = testFF.getWeights()
    for i in range(len(trainY)):
          testFF.learn(train[i], [trainY[i]])
    newWeights = testFF.getWeights()
    wDiff = map(sub, newWeights, oldWeights)
    if np.max(np.abs(wDiff)) < tol:
        converged = iteration + 1
        break

if converged > 0:
    print "Converged after ", converged, "iterations"
else:
    print "Failed to converge after", nIter, "iterations"

resultsTrained = np.array([testFF.feedforward(x)[0] for x in trainBig])
mseTrained = mse(resultsTrained, trainBigY)

for i in range(200):
     testFF.learn([1,1],[10])
     testFF.learn([0,0],[-15])
     testFF.learn([-1,.2],[-20])
     testFF.feedforward([1,1])
     testFF.feedforward([0,0])
     testFF.feedforward([-1,.2])

'''
