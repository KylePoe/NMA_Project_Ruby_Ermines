#!/usr/env/bin python

# From https://github.com/NeuromatchAcademy/course-content/blob/master/projects/load_stringer_spontaneous.ipynb
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from Code.sampling import sample_uniform

mpl.use('Qt5Agg') # Comment out if not in Windows

# fname = "Data/stringer_spontaneous.npy"
fname = "Data/stringer_orientations.npy"
dat = np.load(fname, allow_pickle=True).item()
print(dat.keys())

# 3d scatter plot
scatterplot = plt.figure()
ax = scatterplot.add_subplot(111, projection='3d')

sample = sample_uniform(dat['xyz'], p=0.05)
ax.scatter(sample[0,:], sample[1,:], sample[2,:], c='r', marker='o', alpha=0.1)
plt.show()
