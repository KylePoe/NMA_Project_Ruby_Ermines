#!/usr/env/bin python

# From https://github.com/NeuromatchAcademy/course-content/blob/master/projects/load_stringer_spontaneous.ipynb
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from Code.sampling import sample_uniform, sample_around_point

mpl.use('Qt5Agg') # Comment out if not in Windows

# fname = "Data/stringer_spontaneous.npy"
fname = "Data/stringer_orientations.npy"
dat = np.load(fname, allow_pickle=True).item()
print(dat.keys())

# 3d scatter plot
scatterplot = plt.figure()
ax = scatterplot.add_subplot(111, projection='3d')

# sample = sample_uniform(dat['xyz'], p=0.05)
sample = sample_around_point(
    dat['xyz'],             # Our sample will consist of 3d coordinates for each neuron
    dat['xyz'],             # This is the coordinate data for the neurons
    p=0.1,                  # This is the percentage of the total population we want contained in the sample
    point=[563, 308, -299], # This is the point in 3d space we want to sample around
    cov= np.array([           # Variance of a elliptical gaussian
        [150, 0,0],
        [0, 20, 0],
        [0, 0, 50]]
    )
) # warning! this is a bit computationally expensive...


ax.scatter(sample[0,:], sample[1,:], sample[2,:], c='r', marker='o', alpha=0.1)
ax.set_xlim(6, 1126)
ax.set_ylim(6.65, 640)
ax.set_zlim(-450, -150)
plt.show()
