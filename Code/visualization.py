import matplotlib.pyplot as plt
import matplotlib as mpl


def visualize_voronoi_tesselation(tesselation):

    # Initialize interactive backend
    mpl.use('Qt5Agg')

    # 3d scatter plot
    scatterplot = plt.figure()
    ax = scatterplot.add_subplot(111, projection='3d')

    for i, neuron_group in enumerate(tesselation):
        ax.scatter(neuron_group[0, :], neuron_group[1, :], neuron_group[2, :], marker='o', alpha=0.1)

    plt.show()

def scatterplot(neurons, **kwargs):

    # Initialize interactive backend
    mpl.use('Qt5Agg')

    # 3d scatter plot
    scatterplot = plt.figure()
    ax = scatterplot.add_subplot(111, projection='3d')

    ax.scatter(neurons[0, :], neurons[1, :], neurons[2, :], **kwargs)
    plt.show()
