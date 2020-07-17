#!/usr/env/bin python

import numpy as np

SPONTANEOUS_DATA = "Data/stringer_spontaneous.npy"

# Spontaneous data dimensionality analysis

data = np.load(SPONTANEOUS_DATA, allow_pickle=True).item()

pass