###
### Still need to add an input for take different files
###

import pandas as pd
import numpy as np
import sys

file = sys.argv[1]
    
data = pd.read_csv(file, header=-1)

name = file.split(".")
arff = open(name[0]+".arff",'w')

arff.write("@RELATION " + name[0] + "\n\n")

for col in range(0,len(data.columns)):

    if data.dtypes[col].kind == 'O':
        levels = pd.unique(data[col])
        levels.shape = (len(levels), 1)
        levels = levels.T

        arff.write("@ATTRIBUTE " + "feature_" + str(col) + " " + "{")
        np.savetxt(arff, levels, delimiter = ', ', fmt = '%s', newline = '')
        arff.write("}\n")

    else:
        data_col = "NUMERIC"

arff.write("\n@DATA\n")

np.savetxt(arff, data, delimiter = ', ', fmt = '%s')

arff.close()
