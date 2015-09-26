# generate data for experiments

import numpy as np
import itertools

def rosen(input):
    return sum((1-input[:-1])**2.0 + 100.0*(input[1:]-input[:-1]**2.0)**2.0)

dim = 3
pointsPerDim = 3
width = 3.

# grid sample
# given by np.array([x1], [x2], ...)

oneDim = np.linspace(-width, width, pointsPerDim)

sample = np.meshgrid(*[oneDim for i in range(dim)])
sample = np.swapaxes(sample, 0, dim)
sample = sample.reshape(pointsPerDim**dim, dim)
sample = np.swapaxes(sample, 0, 1)

# uniform sample
#sample = np.random.random_sample((dim, pointsPerDim ** dim))

response = rosen(sample)
