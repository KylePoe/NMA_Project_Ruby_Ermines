#!/usr/env/bin python

# From https://github.com/NeuromatchAcademy/course-content/blob/master/projects/load_stringer_spontaneous.ipynb
import numpy as np

if __name__ == '__main__':
    from download_data import download_data
else:
    from .download_data import download_data


download_data()

fname = "Data/stringer_spontaneous.npy"
dat = np.load(fname, allow_pickle=True).item()
print(dat.keys())