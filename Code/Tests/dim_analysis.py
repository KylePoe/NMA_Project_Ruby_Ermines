from Code.sampling import random_partition, sample_around_point
from Code.file_io import load_spontaneous, load_orientations
import numpy as np
from sklearn.decomposition import PCA
from datetime import datetime
from matplotlib import pyplot as plt

data = load_spontaneous()

AUTOCORRELATION_WINDOW = 4000

number_neurons = 1000

trials_per_size = 30

bootstrap_size = 30

min_sample = 500
max_sample = 11000

N_sizes = 20

N_samples = data['sresp'].shape[1]

# # Perform k-fold sampling on the dimensionality curve
# n_bins = random_partition(data['sresp'].T, n=number_neurons)

data_mat = np.empty((N_sizes, trials_per_size, bootstrap_size))

for isample, sample_size in enumerate(np.linspace(min_sample, max_sample, N_sizes).astype('uint8')):
    print(f'Analyzing sample {isample}...')
    for ibin in range(trials_per_size):
        print(f'\tAnalyzing bin {ibin}...')

        sample = sample_around_point(data['sresp'].T, data['xyz'], v=100, n=sample_size)

        # Do bootstrapping
        for jtime, time in enumerate(np.linspace(0, N_samples - AUTOCORRELATION_WINDOW, bootstrap_size)):
            print(f'\t\tAnalyzing time range {jtime}...')
            time = time.astype('uint8')
            neuron_interval = sample[time:(time + AUTOCORRELATION_WINDOW), :]
            data_pca = PCA(n_components=0.8, svd_solver='full').fit(neuron_interval)
            data_mat[isample, ibin, jtime] = data_pca.components_.shape[0]

np.save(f'dim_analysis_{datetime.now()}.npy')



