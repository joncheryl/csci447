# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 09:34:59 2015

"""

import pandas as pd
import numpy as np
import os

# Set some Pandas options
pd.set_option('html', False)
pd.set_option('max_columns', 30)
pd.set_option('max_rows', 20)

os.chdir("/home/ds0/School/Fall 15/CSCI 447 Machine Learning and Soft Computing/Project 1/")
data = pd.read_csv("abalone.data", low_memory=False)

name = "abalone.data".split(".")
arff = open(name[0]+".arff",'w')

arff.write("@RELATION " + name[0] + "\n\n")

for data_ix in range(0,len(data.columns)):
    if data.iloc[:,data_ix].dtype.char == 'd' or data.iloc[:,data_ix].dtype.char == 'l':
        data_col = "NUMERIC"
    else:
        data_col = "{" + ",".join(unique(data.iloc[:,data_ix])) + "}"
        
    arff.write("@ATTRIBUTE " + "feature_" + str(data_ix) + " " + data_col + "\n")
    
arff.write("\n@DATA\n")

for row in range(0,len(data)):
    arff.write(", ".join(str(x) for x in data.ix[row,]) + "\n")

arff.close()