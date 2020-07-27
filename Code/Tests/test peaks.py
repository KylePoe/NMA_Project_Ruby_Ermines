import os, requests

import scipy

fname = "stringer_spontaneous.npy"
url = "https://osf.io/dpqaj/download"

if not os.path.isfile(fname):
  try:
    r = requests.get(url)
  except requests.ConnectionError:
    print("!!! Failed to download data !!!")
  else:
    if r.status_code != requests.codes.ok:
      print("!!! Failed to download data !!!")
    else:
      with open(fname, "wb") as fid:
        fid.write(r.content)


data = {
    'Orientation': {
        'url': "https://osf.io/ny4ut/download",
        'file': "Data/stringer_orientations.npy"
    },
    'Spontaneous': {
        'url': "https://osf.io/dpqaj/download",
        'file': "Data/stringer_spontaneous.npy"
    }
}

import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

import scipy
import scipy.signal
from scipy.signal import find_peaks

data = np.load('stringer_spontaneous.npy', allow_pickle=True).item()

print(data['pupilArea'].shape)

pupil = np.array(data['pupilArea']).reshape(7018, )
scipy.signal.find_peaks(pupil, height=None, threshold=1000, distance=None, prominence=None, width=None,
                        wlen=None, rel_height=0.5, plateau_size=None)

peaks, _ = find_peaks(pupil, height=0)
plt.plot(pupil)
plt.plot(peaks, pupil[peaks], "var")
plt.plot(np.zeros_like(pupil), "--", color="gray")
plt.show()

pupil1D = np.ndarray.flatten(pupil)
print(pupil1D.shape())
