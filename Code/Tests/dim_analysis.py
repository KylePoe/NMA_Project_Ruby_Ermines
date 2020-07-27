from Code.sampling import random_partition, sample_around_point
from Code.file_io import load_spontaneous, load_orientations
import numpy as np
from sklearn.decomposition import PCA
from datetime import datetime

params = {
    'bootstrap window': 4000,
    'number_neurons': 1000,
    'trials_per_size': 30,
    'bootstrap_size': 30,
    'min_sample': 500,
    'max_sample': 11000,
    'N_sizes': 20,
    'sampling_variance': 100
}

data = load_spontaneous()

N_samples = data['sresp'].shape[1]

# # Perform k-fold sampling on the dimensionality curve
# n_bins = random_partition(data['sresp'].T, n=number_neurons)

data_mat = np.empty((
    params['N_sizes'],
    params['trials_per_size'],
    params['bootstrap_size']
))

for isample, sample_size in enumerate(
        np.linspace(
            params['min_sample'],
            params['max_sample'],
            params['N_sizes']
        ).astype('uint8')
):
    print(f'Analyzing sample {isample}...')
    for ibin in range(params['trials_per_size']):
        print(f'\tAnalyzing bin {ibin}...')

        # Change to cylindrical sampling
        sample = sample_around_point(
            data['sresp'].T,
            data['xyz'],
            v=100,
            n=sample_size
        )

        # Do bootstrapping
        for jtime, time in enumerate(
                np.linspace(
                    0,
                    N_samples - params['bootstrap window'],
                    params['bootstrap_size']
                )
        ):
            print(f'\t\tAnalyzing time range {jtime}...')
            time = time.astype('uint8')
            neuron_interval = sample[time:(time + params['bootstrap window']), :]
            data_pca = PCA(n_components=0.8, svd_solver='full').fit(neuron_interval)
            data_mat[isample, ibin, jtime] = data_pca.components_.shape[0]

np.save(
    f'dim_analysis_{datetime.now()}.npy',
    {
        'parameters': params,
        'output': data_mat
    }
)


