
#parameters: 
def muPlusLambda(fitnessFunc, population, numberOfChildren):
	#set generation counter
	t = 0 
	#number of individuals in population
	mu = population.shape[0] 
	fitness = np.zeros(shape=mu+numberOfChildren)

	for individual in population: #evaluate the fitness
		fitness.append(evaluateFitness(newOffspring))

	notDone = true 
	
	while (t < generationThreshold): 
		for i in range(0, lambda):
			#choose the parents randomly, p=number of parents
			import random
			parents = random.sample(population,p)
			#create the offspring with crossover
			newOffspring = crossover(parents)

			newPopulation.append(newOffspring)
			#evaluate fitness for offspring
			fitness.append(evaluateFitness(newOffspring))
		
		#add all mu to the population as well
		for i in range (0, mu):
			newPopulation.append(population[i])
		#get the indices of the top mu fitnesses
		topFitness = fitness.argsort()[-mu:]
		#set the population to now be the top mu from mu+lambda
		population = newPopulation[[topFitness],:]
		#set the fitnesses back to zero
		fitness = np.zeros(shape=mu+numberOfChildren)
		t++

 
