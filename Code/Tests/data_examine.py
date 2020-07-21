#!/usr/env/bin python

# From https://github.com/NeuromatchAcademy/course-content/blob/master/projects/load_stringer_spontaneous.ipynb
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from Code.sampling import sample_uniform, sample_around_point
from Code.visualization import scatterplot


fname = "Data/stringer_orientations.npy"
dat = np.load(fname, allow_pickle=True).item()
print(dat.keys())

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

# Visualize in 3d space as scatter plot
scatterplot(sample, c='r', marker='o', alpha=0.1)
