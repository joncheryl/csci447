# -*- coding: utf-8 -*-
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
import ffn

def mse(approx, actual):
    return sum(np.square(actual-approx))/len(actual)
  
class population:
    
    def __init__(self, popSize, topology, method="GA", strategy="monogamy", \
                 tdata = "ttt_num.csv"):
        self.DEBUG = True
        self.popSize = popSize
        self.topology = topology
        self.lrate = 0.05
        self.strategy = strategy
        
        self.nEdges = sum(self.topology * append(self.topology[1:],0))
        self.nBiases = sum(self.topology[1:])   
        self.nGenes = self.nEdges + self.nBiases
        self.nSelectFrom = round(self.popSize / 2)  # N parents that reproduce?
        self.nReproduce = round(self.nSelectFrom / 2)  # not used
        
        self.fitnesses = np.empty((popSize, 2))
        
        self.selectFit = np.empty((self.nSelectFrom,2))
        self.selected = np.empty(self.nSelectFrom, self.nGenes)        
        
        self.testSet = pd.read_csv(tdata).as_matrix()        
                
        self.t = 0  # N of iterations of evolution?
        
        # Tuning parameters
        self.mutationRate = 0.3
        self.sigmaMutation = 0.5
        self.xoverProp = 0.5
        self.repProp = 0.8
        self.strategy = strategy
        
        self.tol = 0.90        

        # Initialize population to be uniformly randomly distributed
        a = 0
        b = 1.5
        self.pop = np.random.uniform(a, b, size = (self.popSize, self.nGenes))
        # remove genPop() function 
        #self.genPop()

        self.fitEval()
    
#    def genPop(self):
        # Should probably generate values differently for edges and biases
        #a = 0
        #b = 1.5
        #self.pop = np.random.uniform(a, b, size = (self.popSize, self.nGenes))
        
        #self.fitEval() 

    def fitEval(self):
        if self.DEBUG == True:
            print self.t
        for agent in range(self.popSize):
            weights = self.pop[agent, :self.nEdges]
            biases = self.pop[agent, -self.nBiases:]

            ### dont need to use this old stuff if all we're gonna do is
            ### feedforward
            fitff = ffn.ffNetwork(self.topology, self.lrate, output="binary")
            fitff.setWeights(weights)
            fitff.setBiases(biases)
            
            classifications = np.empty(self.testSet.shape[0])
            classifications.fill(0)
                
            testVal = np.empty(self.topology[-1])
            testVal.fill(0)
            for i in range(self.testSet.shape[0]):
                testVal[self.testSet[i,-1]] = 1
                fitff.learn(list(self.testSet[i,:(self.nEdges + self.nBiases)]), \
                           testVal)
                if np.argmax(fitff.getOutput()) ==  self.testSet[i,-1]:
                   classifications[i] = 1
            if self.DEBUG == True:
                print mean(classifications)                
            self.fitnesses[agent,0] = mean(classifications)
            self.net = fitff
        self.fitnesses[:,1] = list(range(self.popSize))
        
        #fitVals = (self.pop.sum(axis=1) - np.pi)**2
        
        # add fitness values
            
        #self.fitnesses[:,0] = fitVals
        
        # add indices
        
        
        # sort (ranks are implicit after this)
        #self.fitnesses = self.fitnesses[self.fitnesses[:,0].argsort()]

    def mutate(self):
        mutations = np.random.binomial(1, self.mutationRate, self.selected.shape) \
                    * np.random.normal(0, self.sigmaMutation, self.selected.shape) 
        self.selected += mutations 
        
    def crossover(self, parent1, parent2):
        # The
        pLen = len(parent1)
        indices = np.random.choice(pLen, size = pLen * self.xoverProp, \
                  replace = False)
        for j in indices:
            swap = parent1[j]
            parent1[j] = parent2[j]
            parent2[j] = swap
        return parent1, parent2
    
    def select(self):
        self.selectFrom = np.random.choice(self.popSize, self.nSelectFrom, \
                     replace=False)                 
        self.selectFit = self.fitnesses[self.selectFrom]
        self.selected = self.pop[self.selectFrom]
        
        # Rank fitnesses
        rankedSelectFit = self.selectFit[self.selectFit[:,0].argsort()]

        # Decide who can reproduce        
        self.rankedRepFit = rankedSelectFit[0:self.nSelectFrom * self.repProp,:]

        self.bPi = np.random.permutation(self.rankedRepFit)        
        self.breedingPool = self.pop[list(self.bPi[:,1])]
        
        
        
    def pts(self):
        print self.testSet
        
    
    def operators(self):
        # Crossover
        if self.strategy == "monogamy":
            for i in range(int(floor(self.breedingPool.shape[0] / 2))):
                self.breedingPool[i], self.breedingPool[i+2] =  \
                    self.crossover(self.breedingPool[i], self.breedingPool[i+2])
            self.pop[list(pt.bPi[:,1])] = self.breedingPool 
        
        # Mutation
        self.mutate()
        
    def evolve(self):
        while max(self.fitnesses[:,0]) < self.tol:
            print max(self.fitnesses[:,0])            
            self.select()
            self.operators()
            self.t += 1 
            self.fitEval()
            
        return 
        
pt = population(10,[9,18,2])
