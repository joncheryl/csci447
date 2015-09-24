""" 
1) need to add bias nodes
2) make a little test data
3) get backprop working a) first propogate errors back
"""

import numpy as np
import itertools

class Network:

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes

        # Build network topology
        self.topology = [[inputNode(i) for i in range(sizes[0])]]
        for layer in range(len(sizes))[1:]:
            self.topology.append( [Node() for j in range(sizes[layer])] )
            for j in range(sizes[layer]):
                for k in range(sizes[layer-1]):
                    self.topology[layer][j].addInput(self.topology[layer-1][k], np.random.randn(1))

        """        
        for layer in sizes:
            self.hiddenNodes = [Node() for i in range(sizes[1])]
            for i, j in itertools.product(self.hiddenNodes, self.inputNodes):
                i.addInput(j, np.random.randn(1))
                
        self.outputNodes  = [Node() for i in range(sizes[-1])]
        for i, j in itertools.product(self.outputNodes, self.hiddenNodes):
            i.addInput(j, np.random.randn(1))
        """
        
    # input must be a vector of length sizes[0]
    # could be a numpy (n,1) array?
    def feedforward(self, inputVector):
        output = [i.feedforward(inputVector) for i in self.topology[-1]]
        return output

class Node:

    def __init__(self):
        self.input = 0
        self.activation = 0
        self.inputs = []
        
    def addInput(self, otherNode, weight):
        self.inputs.append((otherNode, weight))
        
    def feedforward(self, inputVector):
        self.activation = 0
        for i in self.inputs:
            self.activation += i[0].feedforward(inputVector)*i[1]

        return self.activation

# does this need to inheret from the Node class?
class inputNode(Node):

    def __init__(self, index):
        self.index = index

    def feedforward(self, inputVector):
        return inputVector[self.index]
