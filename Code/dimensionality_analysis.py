#!/usr/env/bin python

import numpy as np

SPONTANEOUS_DATA = "../Data/stringer_spontaneous.npy"

if __name__ == '__main__':

    # Spontaneous data dimensionality analysis
    data = np.load(SPONTANEOUS_DATA, allow_pickle=True).item()

    pass
