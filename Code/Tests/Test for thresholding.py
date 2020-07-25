import numpy as np

def load_spontaneous():
    return np.load("Data/stringer_spontaneous.npy", allow_pickle=True).item()

def load_orientations():
    return np.load("Data/stringer_orientations.npy", allow_pickle=True).item()

import matplotlib.pyplot as plt
import matplotlib as mpl

import scipy
import scipy.signal
from scipy.signal import find_peaks

data = np.load('stringer_spontaneous.npy', allow_pickle=True).item()

print(data['pupilArea'].shape)

pupil = data['pupilArea']

pupil1d = pupil.squeeze()

scipy.signal.find_peaks(pupil1d, height=None, threshold=1000, distance=None, prominence=None, width=None,
                        wlen=None, rel_height=0.5, plateau_size=None)

peaks, _ = find_peaks(pupil1d, height=0)
plt.plot(pupil1d)
plt.plot(peaks, pupil1d[peaks], "var")
plt.plot(np.zeros_like(pupil1d), "--", color="gray")
plt.show()
