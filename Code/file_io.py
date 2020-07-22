import numpy as np

def load_spontaneous():
    return np.load("Data/stringer_spontaneous.npy", allow_pickle=True).item()

def load_orientations():
    return np.load("Data/stringer_orientations.npy", allow_pickle=True).item()
