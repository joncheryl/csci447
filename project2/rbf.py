import numpy as np
import scipy as sp
class RBF:
	def rbfModel(inSet, output, centers):
		#2d array to hold the kmeans 
		km = np.zeros((2, centers))

		#number of observations of the input dataset
		N = len(inSet)
		
		#number of variables of the input dataset
		nCol = len(inSet[0])

		#'Unsupervised' Phase of training:
		#calculate the cluster points, mus, using kmeans
		mus = sp.cluster.kmeans(inSet, km);
		
		#calculate the Euclidean distances
		distances = (mus-mus)**2
		distances = distances.sum(axis=-1)
		distances = np.sqrt(distances)
		
		#calculate sigma
		sigma = max(distances)/sqrt(2*centers)

		#set up phi to be an empty N x K+1 matrix 
		phi = np.zeros((N,centers+1))
		for row in range(0,N):
			#set to bias column to 1
			phi[row,1] = 1
			for col in range (0,centers):
				#set the weights
				phi[row,col+1] = exp(-(1/(2*sigma))*np.linalg.norm(inSet[row:]-mus[col,])**2)
		#set the weights
		weights = np.linalg.pinv(phi.transpose()*phi) * phi.transpose() * output
		return ReturnVal(sigma, weights, mus)

	def rbfPrediction(rbfModel, X):
		sigma = rbfModel.sigma
		centers = rbfModel.centers
		weights = rbfModel.weights
		N = X.shape[0]

		prediction = np.full(N, weights[0])

		for j in range(0, N):
			for k in range (0, len(centers)):
				prediction[j] = prediction[j] + weights[k+1]*exp(-(1/(2*sigma))*np.linalg.norm(X[j:]-centers[k,])**2)
		return prediction

class ReturnVal(object):
	def __init__(self, sigma, weights, centers):
		self.sigma = sigma
		self.weights = weights
		self.centers = centers




