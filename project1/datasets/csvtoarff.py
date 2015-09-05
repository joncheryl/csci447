###
### for f in *; do python csvtoarff.py $f; done
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

    arff.write("@ATTRIBUTE " + "feature_" + str(col) + " ")

    # if clearly an object or less than 10 levels
    if data.dtypes[col].kind == 'O' or len(pd.unique(data[col])) < 10:
        levels = pd.unique(data[col])
        levels.shape = (len(levels), 1)
        levels = levels.T

        arff.write("{")
        np.savetxt(arff, levels, delimiter = ', ', fmt = '%s', newline = '')
        arff.write("}\n")

    else:
        arff.write("NUMERIC\n")

arff.write("\n@DATA\n")

np.savetxt(arff, data, delimiter = ', ', fmt = '%s')

arff.close()
