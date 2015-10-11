'''
Feedforward Neural Network
'''

import numpy as np
import math

def actFun(x, act):
    if act == 0: # Use sigmoid activation function
        if x>0:
            return 1 / (1 + math.exp(-x))
        else:
            return math.exp(x) / (math.exp(x) + 1)
    else: # Use linear activation function
        return slope * x

def dActFun(x, act):
    if act == 0: # Use sigmoid activation function
        return math.exp(-x)/(1+math.exp(-x))**2
    else: # Use linear activation function
        return slope
     
class ffNetwork:

    def __init__(self, sizes, learningRate, momentum = 0, act = 0):
        '''
        Each element in sizes contains the number of nodes in each
        respective layer. For example, if sizes is of length 3, sizes[0] would
        be how many input nodes are specified, sizes[1] is how many nodes are
        in the single hidden layer and sizes[2] is the number of output nodes

        Momentum is disabled by default.

        The default activation function is the sigmoid.
        '''

        self.sizes = sizes
        self.a = learningRate
        self.mom = momentum

        #
        # Build network topology
        #
        # input layer
        self.nodes = [[inputNode(i) for i in range(sizes[0])]]
        self.edges = []

        # hidden layers
        for layer in range(len(sizes))[1:-1]:
            self.nodes.append( [Node(act) for j in range(sizes[layer])] )
            for j in range(sizes[layer]):
                for k in range(sizes[layer-1]):
                    self.connect(self.nodes[layer-1][k], self.nodes[layer][j], np.random.randn(1))

        # output layer
        self.nodes.append([outputNode(i) for i in range(sizes[-1])])
        for j in range(sizes[-1]):
            for k in range(sizes[-2]):
                self.connect(self.nodes[-2][k], self.nodes[-1][j], np.random.randn(1))

    #
    # Method for building network (creating edges and connecting nodes)
    #
    def connect(self, upstream, downstream, weight):
        connector = Edge(upstream, downstream, weight, self.a)
        self.edges += [connector]
        upstream.addOutput(connector)
        downstream.addInput(connector)

    #
    # Feed input through network populating nodes with information
    #
    # input must be a vector of length sizes[0]
    def feedforward(self, inputVector):
        self.output = [i.feedforward(inputVector) for i in self.nodes[-1]]
        return self.output

    #
    # Update weights and biases in network by backpropogation
    #
    def learn(self, inputVector, target):
        # Compute outputs for each node
        self.feedforward(inputVector)

        # Backpropogate errors
        for inputNode in self.nodes[0]:
            for edge in inputNode.outputs:
                edge.down.backProp(target)

        # Update weights
        for edge in self.edges:
            #w = w - learningRate * (dEdOutput * dOutputdInput) * dInputdWeight
            weightChange = edge.a * edge.down.delta * edge.up.output
            edge.weight -= weightChange - edge.oldWeightChange * self.mom
            edge.oldWeightChange = weightChange

        # Update biases
        for layer in range(len(self.sizes))[1:]:
            for node in self.nodes[layer]:
                # w_b = w_b - learningRate * (dEdOutput * dOutputdInput) * 1
                biasChange = node.delta
                node.bias -= self.a * biasChange - node.oldBiasChange * self.mom
                node.oldBiasChange = biasChange

    #
    # Methods to obtain weights and biases for averaging during the training process
    #
    def getWeights(self):
        weights=[]
        for edge in self.edges:
            weights.append(edge.weight[0])
        return(weights)
    
    def getBiases(self):
        biases=[]
        for node in self.nodes:
            node.append(node.bias[0])
        retuern(biases)
        
class Edge:
    def __init__(self, upstream, downstream, weight, learningRate):
        # upstream and downstream nodes
        self.up = upstream
        self.down = downstream

        self.weight = weight
        self.oldWeightChange = 0
        '''
        having the learning rate defined here may potentially allow for
        different learning rates at different levels of the network. May be
        interesting to test, time permitting
        '''
        self.a = learningRate

    def getOutput(self, inputVector):
        return self.up.feedforward(inputVector) * self.weight
                           
class Node:
    def __init__(self, act):
        self.act = act
        self.netInput = 0
        self.output = 0
        self.inputs = []
        self.outputs = []
        
        self.bias = np.random.randn(1)
        self.oldBiasChange = 0
        self.delta = 0
        
    def addInput(self, edge):
        self.inputs.append(edge)

    def addOutput(self, edge):
        self.outputs.append(edge)
        
    def feedforward(self, inputVector):
        self.netInput = 0 + self.bias
        for i in self.inputs:
            self.netInput += i.getOutput(inputVector)

        self.output = actFun(self.netInput, self.act)
        return self.output

    def backProp(self, target):
        self.delta = 0

        # dEdOutput
        for outEdge in self.outputs:
            self.delta += outEdge.down.backProp(target) * outEdge.weight

        # dOutputdInput
        self.delta *= dActFun(self.netInput, self.act)

        return self.delta

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
        self.oldBiasChange = 0
        self.delta = 0
        
    def addInput(self, edge):
        self.inputs.append(edge)

    def feedforward(self, inputVector):
        self.netInput = 0 + self.bias
        for i in self.inputs:
            self.netInput += i.getOutput(inputVector)

        self.output = self.netInput
        return self.output[0]

    def backProp(self, target):
        # dEdOutput
        self.delta = (self.output - target[self.index])

        # dOutputdInput = 1 so nothing to multiply
        return self.delta
