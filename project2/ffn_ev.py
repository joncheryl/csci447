# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from operator import sub
import ffn

def rosen(x):
     """The Rosenbrock function"""
     return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)
     
def mse(approx, actual):
    return sum(np.square(actual-approx))/len(actual)

def absErr(approx, actual):
    return sum(np.abs(actual-approx))/len(actual)
    
def partition(df, npart):
    parts = [0]
    partInc = int(np.floor(len(df.index)/npart))
    
    if partInc == 0:
        partInc = 1
    dfLen = len(df.index)
    
    for i in range(dfLen):
        part = partInc*(i+1)
        if part > dfLen  or len(parts) >= npart:            
            parts.append(dfLen)
            return(parts)
        else:
            parts.append(part)
            
    parts.append(dfLen)
    return parts        
    
def kfold(df, k, niter=10000, lrate= 0.05, tol = 0.001, topology = [2,4,1], 
          ndim = 2, momentum = 0):
    
    errs = pd.DataFrame()

    parts = partition(df, k)
    for j in range(len(parts) - 1):
        converged = 0

        testData =  df.ix[parts[j]:parts[j+1]-1,]
        trainData = df[df.index.isin(testData.index) == False]
        
        rosff = ffn.ffNetwork(topology, lrate, momentum)
        for i in range(len(trainData.index)):
        #for i in range(niter):
            oldWeights = rosff.getWeights()
            #example = trainData.sample()            
            rosff.learn(trainData.iloc[i][0:ndim],[trainData.iloc[i][ndim]])
            newWeights = rosff.getWeights()
            wDiff = map(sub, newWeights, oldWeights)
            if np.max(np.abs(wDiff)) < tol:
                converged = i + 1
#        break
        if converged > 0:
            print "Fold ", j+1, " converged after ", converged, "iterations"
        else:
            print "Fold ", j+1, " failed to converge after", len(trainData.index), "iterations"
            
        rosVals = np.array([])
        ffVals = np.array([])
        
        for l in range(len(testData.index)):
            rosff.learn(testData.iloc[l][0:ndim],[testData.iloc[l][ndim]])
            rosVals = np.append(rosVals, testData.iloc[l][ndim])
            ffVals = np.append(ffVals, rosff.output[0])
            
        MSE = mse(ffVals,rosVals)
        RMS = np.sqrt(MSE)        
        mAbsErr = absErr(ffVals,rosVals)           
        correlation = np.corrcoef(rosVals,ffVals)[1][0]
        if np.isnan(correlation):
            correlation = 0
        errs = errs.append([[MSE, RMS, mAbsErr, correlation]])
    errs.columns = ['MSE','RMS','Absolute','Correlation']   
    errs.index = np.add(range(k),1)    
    print errs
    print "\n%d-fold cross-validation averaged errors:" % k
    print errs.mean() 
    return errs.mean()

def ten_fold(df):
    ndim = len(df.columns) - 1
    ffn_ev.kfold(df, 10, topology = [ndim, 2*ndim, 1], ndim=ndim)

n2 = pd.read_csv("n2.csv",header=None)
n3 = pd.read_csv("n3.csv",header=None)
n4 = pd.read_csv("n5.csv",header=None)
n5 = pd.read_csv("n5.csv",header=None)
n6 = pd.read_csv("n6.csv",header=None)

ten_fold(n2)
