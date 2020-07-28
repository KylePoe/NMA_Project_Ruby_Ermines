import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

def visualize_voronoi_tesselation(tesselation):

    # Initialize interactive backend
    mpl.use('Qt5Agg')

    # 3d scatter plot
    scatterplot = plt.figure()
    ax = scatterplot.add_subplot(111, projection='3d')

    for i, neuron_group in enumerate(tesselation):
        ax.scatter(neuron_group[0, :], neuron_group[1, :], neuron_group[2, :], marker='o', alpha=0.1)

    plt.show()
    return scatterplot


def scatterplot(neuron_locs, proj_z: bool = False, **kwargs):

    # Initialize interactive backend
    mpl.use('Qt5Agg')

    if 'axis' in kwargs:
        scatterplot = kwargs['axis']
    else:
        scatterplot = plt.figure()

    # 3d scatter plot
    if not proj_z:
        ax = scatterplot.add_subplot(111, projection='3d')
        ax.scatter(neuron_locs[0, :], neuron_locs[1, :], neuron_locs[2, :], **kwargs)
    else:
        ax = scatterplot.add_subplot(111)
        ax.scatter(neuron_locs[0, :], neuron_locs[1, :], **kwargs)

    plt.show()
    return scatterplot

def visualize_dim_analysis(file):
    data = np.load(file, allow_pickle=True).item()

