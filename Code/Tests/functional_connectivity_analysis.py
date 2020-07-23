from Code.file_io import load_spontaneous
import numpy as np
from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, leaves_list

data = load_spontaneous()
neurons = data['sresp']

n_neurons, m_samples = neurons.shape

# Get covariance
cov = (neurons.T - neurons.mean(axis=1)).T @ (neurons.T - neurons.mean(axis=1))/ m_samples
pearson_correlation = cov / np.outer(np.std(neurons, axis=1), np.std(neurons, axis=1))

# Compute network clustering (Code snippet from https://github.com/nilearn/nilearn/blob/master/nilearn/plotting/matrix_plotting.py)
linkage_matrix = linkage(pearson_correlation, method='average')
ordered_linkage = optimal_leaf_ordering(linkage_matrix, pearson_correlation)
index = leaves_list(ordered_linkage)
connectivity_matrix = pearson_correlation[index, :][:, index]

# Save to file
np.save(
    f'Data/connectivity_analysis_spontaneous.npy', {
        'Connectivity Matrix': connectivity_matrix,
        'Index': index,
        'Ordered Linkage': ordered_linkage,
        'Linkage Matrix': linkage_matrix
    }
)
