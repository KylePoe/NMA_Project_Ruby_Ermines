from Code.sampling import random_partition, sample_around_point, sample_cylindrical, sample_uniform
from Code.file_io import load_spontaneous, load_orientations
import numpy as np
from sklearn.decomposition import PCA
from datetime import datetime
import timeit



def dim_analysis(params, fname=None):
    if params['dataset'] == 'spontaneous':
        data = load_spontaneous()
    else:
        data = load_orientations()

    N_samples = data['sresp'].shape[1]

    data_mat = np.empty((
        params['N_sizes'],
        params['trials_per_size'],
        params['bootstrap_size']
    ))

    start = timeit.default_timer()
    sample_sizes = np.linspace(
        params['min_sample'],
        params['max_sample'],
        params['N_sizes']
    ).astype('int')

    for isample, sample_size in enumerate(sample_sizes):
        print(f'Analyzing sample {isample}...')
        for ibin in range(params['trials_per_size']):
            print(f'\tAnalyzing bin {ibin}', end="")

            if not params['uniform']:
                sample = sample_cylindrical(
                    data['sresp'].T,
                    data['xyz'],
                    v=100,
                    n=sample_size,
                    expand=True
                )
            else:
                sample = sample_uniform(data['sresp'].T, n=sample_size)

            # Do bootstrapping
            for jtime, time in enumerate(
                    np.linspace(
                        0,
                        N_samples - params['bootstrap window'],
                        params['bootstrap_size']
                    ).astype('int')
            ):
                # print(f'\t\tAnalyzing time range {jtime}...')
                print('.', end="")
                neuron_interval = sample[time:(time + params['bootstrap window']), :]
                data_pca = PCA(n_components=0.8, svd_solver='full').fit(
                    (neuron_interval - neuron_interval.mean(axis=0))/neuron_interval.std(axis=0)
                )
                data_mat[isample, ibin, jtime] = data_pca.components_.shape[0]

            print('\n', end="")

    stop = timeit.default_timer()
    print(f'...Done in {stop - start}.')

    if fname is None:
        fname = f'dim_analysis_{datetime.now()}.npy'

    np.save(
        fname,
        {
            'parameters': params,
            'output': data_mat,
            'time elapsed': stop - start,
            'sample sizes': sample_sizes,
        }
    )

if __name__ == '__main__':
    params = {
        'bootstrap window': 1000,
        'trials_per_size': 1,
        'bootstrap_size': 50,
        'min_sample': 11983,
        'max_sample': 11983,
        'N_sizes': 1,
        'sampling_variance': 100,
        'uniform': True,
        'dataset': 'spontaneous'
    }

    dim_analysis(params)
