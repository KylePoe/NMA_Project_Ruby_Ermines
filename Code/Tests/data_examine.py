#!/usr/env/bin python

# From https://github.com/NeuromatchAcademy/course-content/blob/master/projects/load_stringer_spontaneous.ipynb
import numpy as np
from Code.sampling import sample_uniform, sample_around_point, get_layer, random_partition
from Code.visualization import scatterplot
from Code.file_io import load_spontaneous

dat = load_spontaneous()
print(dat.keys())

# Random partition
part = random_partition(dat['xyz'], n=50)


# sample = sample_uniform(dat['xyz'], p=0.05)
# sample = sample_around_point(
#     dat['xyz'],             # Our sample will consist of 3d coordinates for each neuron
#     dat['xyz'],             # This is the coordinate data for the neurons
#     p=0.1,                  # This is the percentage of the total population we want contained in the sample
#     point=[563, 308, -299], # This is the point in 3d space we want to sample around
#     cov= np.array([           # Variance of a elliptical gaussian
#         [150, 0,0],
#         [0, 20, 0],
#         [0, 0, 50]]
#     )
# ) # warning! this is a bit computationally expensive...

# Visualize in 3d space as scatter plot
# scatterplot(sample, c='r', marker='o', alpha=0.1)

# Visualize one plane
for depth in [-300, -330, -360, -390]:
    layer = get_layer(dat['xyz'], dat['xyz'], depth=depth, return_closest=True)
    scatter = scatterplot(layer, proj_z=True, c='b', marker='o', alpha=0.4)
    scatter.suptitle(depth)

plane_sample = sample_around_point(
    layer,
    layer,
    p=0.4,
    point=[500, 300],
    v=100
)

scatterplot(plane_sample, c='r', marker='o', alpha=0.4)
