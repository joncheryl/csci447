"""
Evolve()
   t <- 0;
   p(t) initialize();
   f(t) <- evaluate(P(t))
   While (not terminate(P())) do
      t <- t + 1;
      C(t) <- select(P(t-1),f(t));
      C'(t) <- operators(C(t));
      f'(t) <- evaluate(c'(t));
      P(t) <- replace(C'(t),f'(t));
   end while;
end;

biases: 1 for each hidden, and for output
weights: nxm matrix for n nodes going to m nodes

"""

import numpy as np
import pandas as pd
#import ffn  # not used


def mse(approx, actual):
    return sum(np.square(actual-approx))/len(actual)


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


class population:
    def __init__(self, popSize, topology, tdata,
                 method="GA", strategy="monogamy"):
        self.DEBUG = True
        self.lrate = 0.05  # not used
        self.testSet = pd.read_csv(tdata).as_matrix()

        # Tuning parameters
        self.popSize = popSize
        self.mutationRate = 0.3
        self.sigmaMutation = 0.5
        self.xoverProp = 0.5
        self.repProp = 0.8
        self.strategy = strategy
        self.t = 0  # N of iterations of evolution?
        self.tol = 0.90

        # topology parameters
        self.topology = np.array(topology)
        self.nEdges = sum(self.topology[:-1] * self.topology[1:])
        self.nBiases = sum(self.topology[1:])
        # to organize topology for feedforward (by layer)
        self.nWLayer = self.topology[1:] * self.topology[:-1]
        self.cumW = np.append([0], np.cumsum(self.nWLayer))
        self.cumB = np.append([0], np.cumsum(topology[1:]))

        # geneticy parameters
        self.nGenes = self.nEdges + self.nBiases
        self.nSelectFrom = round(self.popSize / 2)  # N parents that reproduce?
        self.nReproduce = round(self.nSelectFrom / 2)  # not used
        # do these need to be initialized?
        self.fitnesses = np.zeros((popSize, 2))
        self.selectFit = np.zeros((self.nSelectFrom, 2))
        self.selected = np.zeros((self.nSelectFrom, self.nGenes))

        # diffEvo parameters
        self.CR = .5  # crossover probability (between 0 and 1)
        self.F = 1  # differential weight (between 0 and 2)
        
        # Initialize population to be uniformly randomly distributed
        self.leftEndpoint = -1
        self.rightEndpoint = 1

        # each row represents a different agent
        # the first nEdges elements in each row are the weights of the network
        # the last nBiases elements in each row are the biases of the network
        self.pop = np.random.uniform(self.leftEndpoint, self.rightEndpoint,
                                     size=(self.popSize, self.nGenes))

        self.fitEval()

    #
    # Evaluate fitness of population
    #
    def fitEval(self):
        if self.DEBUG is True:
            print(self.t)

        # compute fitness of each agent in population
        for agent in range(self.popSize):
            # make a network that each agent represents
            weights = self.pop[agent, :self.nEdges]
            wVectors = [weights[i:j] for i, j in
                        zip(self.cumW[:-1], self.cumW[1:])]
            weights = [wVectors[i].reshape(self.topology[i + 1],
                                           self.topology[i])
                       for i in range(len(self.topology) - 1)]
            biases = self.pop[agent, -self.nBiases:]
            biases = [biases[i:j] for i, j in
                      zip(self.cumB[:-1], self.cumB[1:])]

            # vector of classifications (0 if incorrect, 1 if correct)
            classifications = np.zeros(self.testSet.shape[0])

            # classify test points given agent
            for i in range(self.testSet.shape[0]):
                # feedforward
                a = self.testSet[i, :-1]

                for w, b in zip(weights, biases):
                    a = sigmoid(np.dot(w, a) + b)

                # if correctly classified...
                if (
                        len(np.unique(a)) == len(a) and
                        np.argmax(a) == self.testSet[i, -1]
                ):
                    classifications[i] = 1

#            if self.DEBUG is True:
#                print(np.mean(classifications))

            # make the fitness of agent the average classification rate
            self.fitnesses[agent, 0] = np.mean(classifications)

        # append population member ?
        self.fitnesses[:, 1] = range(self.popSize)

        # nfitVals = (self.pop.sum(axis=1) - np.pi)**2
        # add fitness values

        # self.fitnesses[:,0] = fitVals

        # add indices

        # sort (ranks are implicit after this)
        # self.fitnesses = self.fitnesses[self.fitnesses[:,0].argsort()]

    #
    # Determine which agents are selected to reproduce
    #
    def select(self):
        # select pool of agents to be ranked
        self.selectFrom = np.random.choice(self.popSize, self.nSelectFrom,
                                           replace=False)
        self.selectFit = self.fitnesses[self.selectFrom]
        self.selected = self.pop[self.selectFrom]

        # rank selected agents by fitnesses
        rankedSelectFit = self.selectFit[self.selectFit[:, 0].argsort()]

        # decide who can reproduce
        nCanReproduce = self.nSelectFrom * self.repProp
        self.rankedRepFit = rankedSelectFit[0:nCanReproduce, :]

        # randomly shuffle selected agents to create breeding pool
        self.bPi = np.random.permutation(self.rankedRepFit)
        self.breedingPool = self.pop[list(self.bPi[:, 1])]

    #
    # Mutate selected agents in population by random amounts
    #
    def mutate(self):
        muts = np.random.binomial(1, self.mutationRate, self.selected.shape)
        muts *= np.random.normal(0, self.sigmaMutation, self.selected.shape)
        self.selected += muts

    def operators(self):
        # Crossover
        if self.strategy == "monogamy":
            for i in range(int(floor(self.breedingPool.shape[0] / 2))):
                self.breedingPool[i], self.breedingPool[i+2] =  \
                    self.crossover(self.breedingPool[i], self.breedingPool[i+2])
            self.pop[list(pt.bPi[:,1])] = self.breedingPool 

        # Mutation - maybe should not be put in the operators function
        self.mutate()

    def crossover(self, parent1, parent2):
        # The
        pLen = len(parent1)
        indices = np.random.choice(pLen, size=pLen * self.xoverProp,
                                   replace=False)
        for j in indices:
            swap = parent1[j]
            parent1[j] = parent2[j]
            parent2[j] = swap
        return parent1, parent2

    def evolve(self):
        while max(self.fitnesses[:, 0]) < self.tol:
            print(max(self.fitnesses[:, 0]))
            self.select()
            self.operators()
            self.t += 1
            self.fitEval()

        return

    #
    # Differential Evolution generation step
    # psuedo-code on wiki article
    #
    def diffEvo(self):
        for agent in range(self.popSize):

            # choose unique agents to generate intermediate agent
            friends = [agent]
            while agent in friends:
                friends = np.random.choice(self.popSize, 3, replace=False)

            # make intermediate agent
            z = self.pop[friends[0]] + \
                self.F*(self.pop[friends[1]] - self.pop[friends[2]])

            # combine agent with intermediate agent using crossover prob
            r = np.random.binomial(1, self.CR, self.nGenes)
            r[np.random.choice(self.nGenes, 1)] = 1  # make sure 1 gene changes
            y = (1 - r) * self.pop[agent] + r * z

            # if better, then replace
            if self.fitness(y) > self.fitness(self.pop[agent]):
                self.pop[agent] = y

    #
    # Fitness function for problem
    # Classification rate of agent (the variable agent is like agend ID)
    #
    def fitness(self, agent):
        # make a network that each agent represents
        weights = self.pop[agent, :self.nEdges]
        wVectors = [weights[i:j] for i, j in
                    zip(self.cumW[:-1], self.cumW[1:])]
        weights = [wVectors[i].reshape(self.topology[i + 1],
                                       self.topology[i])
                   for i in range(len(self.topology) - 1)]
        biases = self.pop[agent, -self.nBiases:]
        biases = [biases[i:j] for i, j in
                  zip(self.cumB[:-1], self.cumB[1:])]
        
        # vector of classifications (0 if incorrect, 1 if correct)
        classifications = np.zeros(self.testSet.shape[0])

        # classify test points given agent
        for i in range(self.testSet.shape[0]):
            # feedforward
            a = self.testSet[i, :-1]
            
            for w, b in zip(weights, biases):
                a = sigmoid(np.dot(w, a) + b)
                
            # if correctly classified...
            if (
                    len(np.unique(a)) == len(a) and
                    np.argmax(a) == self.testSet[i, -1]
            ):
                classifications[i] = 1

        # make the fitness of agent the average classification rate
        self.fitnesses[agent, 0] = np.mean(classifications)

#        if self.DEBUG is True:
#            print(np.mean(classifications))

        

pt = population(10, [9, 18, 2], "ttt_num.csv")
print("\n Max: " + str(max(pt.fitnesses[:, 0])))
print("\n Min: " + str(min(pt.fitnesses[:, 0])))

'''
#
# diffEvo script
#

iterations = 10
pt = population(10, [9, 18, 2], "ttt_num.csv")

for i in range(iterations):
    pt.diffEvo()

print("\n Max: " + str(max(pt.fitnesses[:, 0])))
print("\n Min: " + str(min(pt.fitnesses[:, 0])))

'''
