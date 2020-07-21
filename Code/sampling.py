import numpy as np

__author__ = 'Kyle Poe'

"""
Different strategies for sampling neurons that we might want to use.
"""

def sample_uniform(neurons, n=None, p=None):
    """Uniformly sample n neurons or p percent of neurons from the population

    :param neurons: 2D array. Each column should correspond to a single neuron.
    """
    try:
        if n is not None:
            nn = n

        elif p is not None:
            nn = round(neurons.shape[1] * p)
        else:
            raise Exception('Please provide either sample size or percentage')

        return neurons[:, np.random.choice(neurons.shape[1], size=nn, replace=False)]

    except IndexError:
        print('op')
        return sample_uniform(neurons.T, n=n, p=p) # Try again with transpose
