import numpy as np
from scipy.stats import multivariate_normal
from scipy.spatial import cKDTree

__author__ = 'Kyle Poe'

"""
Different strategies for sampling neurons that we might want to use.
"""

def sample_uniform(neurons, n=None, p=None, prob=None):
    """Uniformly sample n neurons or p percent of neurons from the population

    :param p: Percentage of total neuron population to sample
    :param n: Number of neurons to sample
    :param neurons: 2D array. Each column should correspond to a single neuron.
    """

    nn = _get_nn(neurons, n, p)
    return neurons[:, np.random.choice(neurons.shape[1], size=nn, replace=False, p=prob)]


def sample_around_point(neurons, neuron_locs, n=None, p=None, point=None, x=None, y=None, z=None, v=None, cov=None):
    """Draw a normally distributed sample of a particular size centered around a point

    :param neurons: 2D array of neurons, columns correspond to specific neurons.
    :param neuron_locs: 2D array of neuron locations in 3D space. 3xN array
    :param n: number of neurons to sample
    :param p: pedrcentage of neurons to sample
    :param point: point in 3d space to center sampling distribution around
    :param x: x-coordinate of point in 3d space to center sampling distribution around
    :param y: y-coordinate of point in 3d space to center sampling distribution around
    :param z: z-coordinate of point in 3d space to center sampling distribution around
    :param v: variance of gaussian, specifies a spherical distribution
    :param cov: covariance matrix of gaussian for arbitrary orientation
    """

    # Get neurons
    nn = _get_nn(neurons, n, p)

    layer = False

    # Determine if using layer
    if len(np.unique(neurons[2, :])) == 1:
        # Continue adapting to 2d version
        neuron_locs = neuron_locs[(0, 1), :]
        depth = neurons[2, 0]
        layer = True

    if point is not None:
        pp = np.array(point)
    elif layer:
        if x is not None and y is not None:
            pp = np.array([x,y])
        else:
            raise Exception('Please provide x/y or a point')
    else:
        if x is not None and y is not None and z is not None:
            pp = np.array([x,y,z])
        else:
            raise Exception('Please provide x/y/z or a point')

    # Determine covariance matrix
    if cov is not None:
        if not cov.shape == (3 - layer, 3 - layer):
            raise Exception('Covariance matrix dimensions do not match data')

    elif v is not None:
        if layer:
            cov = np.diag(v * np.ones(2))

        else:
            cov = np.diag(v * np.ones(3))
    else:
        raise Exception('You gotta specify the variance')

    probs = multivariate_normal.pdf(neuron_locs.T, mean=pp, cov=cov)
    probs /= sum(probs)

    if nn > sum(probs != 0):
        print(f'WARNING! Asking for too many neurons. Asked for {nn}, max available {sum(probs != 0)}')
        nn = sum(probs != 0)

    return neurons[:, np.random.choice(neurons.shape[1], nn, replace=False, p=probs)]

def voronoi_tesselation(neurons, points):
    voronoi_kdtree = cKDTree(points.T)
    test_point_dist, test_point_regions = voronoi_kdtree.query(neurons.T, k=1)
    return [
        neurons[
            :,
            test_point_regions == iregion           # Return list of heuron groupings
        ] for iregion in range(voronoi_kdtree.n)
    ]

def _get_nn(neurons, n, p):
    if n is not None:
        return n

    elif p is not None:
        return round(neurons.shape[1] * p)
    else:
        raise Exception('Please provide either sample size or percentage')

def get_layer(neurons, neuron_loc, depth=None, return_closest: bool=False):
    """Obtain the layer of neurons corresponding to layer number or specific depth."""

    layers = np.unique(neuron_loc[2, :])

    if depth is not None:
        if depth in layers:
            pass
        elif return_closest:
            depth = layers[np.argmin(np.abs(layers - depth))]
        else:
            raise Exception('Provided depth does not correspond to layer.')

    neuron_mask = neuron_loc[2, :] == depth
    return neurons[:, neuron_mask]

def random_partition(neurons, n=None, p=None):
    pp = _get_nn(neurons, n, p)
    shuffled = np.random.permutation(neurons.T).T
    N = int(neurons.shape[1]/pp)

    return [shuffled[:, (i*pp):(i+1)*pp] for i in range(N)]


def random_interval(n, N):
        """Select random interval of length n from an array of length N"""

        max_start_index = N - n
        start = np.random.random_integers(0, max_start_index)
        return [start, start + n]