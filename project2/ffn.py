'''
still need to get biases to learn
'''

import numpy as np
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x)) if x>0 else math.exp(x) / (math.exp(x) + 1)

def dsigmoid(x):
    return math.exp(-x)/(1+math.exp(-x))**2
    
class Network:

    def __init__(self, sizes, learningRate):
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
        output = [i.feedforward(inputVector) for i in self.nodes[-1]]
        return output

    def backProp(self, inputVector, target):
        self.feedforward(inputVector)
        for inputNode in self.nodes[0]:
            for edge in inputNode.outputs:
                edge.down.backProp(target)

    def learn(self, inputVector, target):
        self.backProp(inputVector, target)
        for edge in self.edges:
            edge.weight -= edge.a * edge.down.delta * edge.up.output
        
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
        
    def getError(self):
        error = 0
        return error
                           
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
        
        self.netInput = 0 + self.bias
        for i in self.inputs:
            self.netInput += i.getOutput(inputVector)

        self.output = sigmoid(self.netInput)
        return self.output

    def backProp(self, target):
        self.delta = 0
        for outEdge in self.outputs:
            self.delta += outEdge.down.backProp(target) * outEdge.weight
        self.delta *= dsigmoid(sum(inEdge.up.output for inEdge in self.inputs))

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
        self.netInput = 0 + self.bias
        for i in self.inputs:
            self.netInput += i.getOutput(inputVector)

        self.output = self.netInput
        return self.output

    def backProp(self, target):
        self.delta = (self.output - target[self.index]) * dsigmoid(sum(inEdge.up.output for inEdge in self.inputs))

        return self.delta
