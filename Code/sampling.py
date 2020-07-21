import numpy as np
from scipy.stats import multivariate_normal

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

    if point is not None:
        pp = np.array(point)

    elif x is not None and y is not None and z is not None:
        pp = np.array([x, y, z])
    else:
        raise Exception('Please provide coordinates')

    # Determine covariance matrix
    if cov is not None:
        pass
    elif v is not None:
        cov = np.diag([v, v, v])
    else:
        raise Exception('You gotta specify the variance')

    probs = multivariate_normal.pdf(neuron_locs.T, mean=pp, cov=cov)
    probs /= sum(probs)

    if nn > sum(probs != 0):
        print(f'WARNING! Asking for too many neurons. Asked for {nn}, max available {sum(probs != 0)}')
        nn = sum(probs != 0)

    return neurons[:, np.random.choice(neurons.shape[1], nn, replace=False, p=probs)]

def _get_nn(neurons, n, p):
    if n is not None:
        return n

    elif p is not None:
        return round(neurons.shape[1] * p)
    else:
        raise Exception('Please provide either sample size or percentage')
