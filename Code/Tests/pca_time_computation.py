from Code.file_io import load_spontaneous
from Code.sampling import random_partition, random_interval
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

data = load_spontaneous()

 # Determine length of neuron timeseries
ts = data['sresp']
N_neurons = ts.shape[0]
M_samples = ts.shape[1]

neuron_bins = 20
sample_bins = 20

neuron_counts = np.rint(
    N_neurons * np.linspace(0.01, 0.2, neuron_bins)
).astype(np.int32)

sample_counts = np.rint(
    M_samples * np.linspace(0.01, 0.2, sample_bins)
).astype(np.int32)

dim_mat = np.zeros((neuron_bins, sample_bins))
std_mat = np.zeros((neuron_bins, sample_bins))

for i, n in enumerate(neuron_counts):
    print(f'Calculating {n} neurons... ({i}/{neuron_bins})')
    for j, m in enumerate(sample_counts):
        print(f'\tCalculating {m} samples... ({j}/{sample_bins})')
        neuron_samples = random_partition(ts.T, n)
        comps = np.zeros_like(neuron_samples)
        for k, sample in enumerate(neuron_samples):
            interval = random_interval(m, sample.shape[0])
            data_pca = PCA(n_components = 0.8, svd_solver='full').fit(sample[interval[0]:interval[1]])
            comps[k] = data_pca.components_.shape[0]

        dim_mat[i, j] = np.mean(comps)
        std_mat[i, j] = np.std(comps)

        print(f'\t\tMean dimension: {dim_mat[i, j]} +/- {std_mat[i, j]}')


nn, ss = np.meshgrid(neuron_counts, sample_counts)
plt.pcolor(nn, ss, dim_mat)
plt.show()


