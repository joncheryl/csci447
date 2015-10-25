'''
Differential Evolution Stategy Implementation
'''

import numpy as np
import pandas as pd


def sigmoid(x):
    if x > 0:
        return 1 / (1 + np.exp(-x))
    else:
        return np.exp(x) / (1 + np.exp(x))


class agent:
    def __init__(self, topology):
        self.num_layers = len(topology)
        self.topology = topology
        self.biases = [np.random.randn(x) for x in topology[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(topology[:-1], topology[1:])]

    def ff(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a


# If only performing a comparison then don't need to return the fitness level
def fitness(agentGuy, inputs, targets):
    mse = 0
    for x, y in zip(inputs, targets):
        mse += (agentGuy.ff(x) - y)**2
    # or maybe
    # mse = sum([ (agentGuy.ff(x) - y)**2 for x, y in zip(inputs, targets) ])

    return mse / len(inputs)

# all agents must be distinct
topo = np.array([2, 4, 1])
nWeights = sum(topo[:-1]*topo[1:])
nBiases = sum(topo[1:])
nDim = nWeights + nBiases
CR = .5  # crossover rate
F = .2  # differential weight


def crossover(agentX, agentA, agentB, agentC):
    index = np.random.randint(0, nDim)

    agentTrail = agent(topo)

    agentTrail.weights = agentA.weights + F*(agentB.weights - agentC.weights)

    for layer in range(len(topo) - 1):  # for every layer
        for x in range(topo[layer]):  # for every input node should put x here
            for y in range(topo[layer+1]):  # for every output node
                if np.random.rand() > CR:
                    agentTrail.weights[layer][x][y] = \
                        agentX.weights[layer][x][y]

    if index > nWeights:
        # do something

    return agentTrail
