import numpy as np
import seaborn as sns
from tqdm import tqdm
from scipy.stats import zscore
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from multiprocessing import Process, Manager, Pool

from Code import sampling
from Code.file_io import load_spontaneous

__author__ = 'Cameron Smith'

"""
Calculates bootstrapped variance explained based on sampling of neurons
"""


def demo_variance_explained_curve(use_multiprocessing=False):
    """ 
    Load example data and creates plot of dimensionality based on sample size
    """
    
    # Load data and calculate (assumes .npy file is in the same directory)
    # neurons = np.load('stringer_spontaneous.npy', allow_pickle=True).item()['sresp']
    neurons = load_spontaneous()['sresp']

    cell_sample_nums = np.arange(10,20)
    cum_var_cutoff = 0.8
    dmeans, dlower, dupper = get_variance_explained_curve(neurons, cell_sample_nums, cum_var_cutoff, use_multiprocessing=use_multiprocessing)
    
    # Plot dimensionality means and confidence intervals
    ax = plt.subplots(1,1,figsize=(10,10))[1]
    ax.plot(cell_sample_nums, dmeans)
    ax.fill_between(cell_sample_nums, (dlower), (dupper), color='b', alpha=.1, label='95%-Confidence Interval')
    plt.plot(cell_sample_nums, dmeans, color='b', label=f'Mean Dimensionality')
    plt.xlabel('Number of Cells Sampled')
    plt.ylabel(f'Dimensionality (Cummulative Var > {int(100*cum_var_cutoff)}%)')
    plt.title('Dimensionality of Spontaneous V1 Activity')
    plt.legend()
    plt.show()
    plt.close()
    return


def get_variance_explained_curve(neurons, cell_sample_nums, cum_var_cutoff=0.8, pca_repetitions=10,
                                sampling_method='sample_uniform', use_multiprocessing=True, **kargs):
    """ Return a curve of variance explained. Extra arguments are passed to the sampling function.
    
    Warnings: 1) Data will be z-score transformed after being passed in, do not preemptively z-score your array.
              2) Returned data will be sorted from lowest to highest cell_sample_nums.
    
    :param neurons: 2D array. Raw data. Each column should correspond to a single neuron.
    :param cell_sample_nums: 1D Int array. Contains sample numbers to use.
    :param cum_var_cutoff: Float. Between 0 and 1. Cutoff for cumulative variance explained.
    :param pca_repetitions: Int. Number of PCA repeats for each sample_num
    :param sampling_method: Str. Unused at this time.
    :param use_multiprocessing: Bool. Set to False if multiprocessing functions throw errors.
    
    Returns three lists: dimensionality means, lower confidence intervals, and upper confidence intervals
    """

    sampling_func_lookup = {'sample_uniform': sampling.sample_uniform}
    
    # This is shuffled to better estimate runtime in TQDM
    shuff_cell_sample_nums = np.copy(cell_sample_nums)
    np.random.shuffle(shuff_cell_sample_nums)

    # Create empty arrays to store values
    dimensionality_means = np.zeros_like(shuff_cell_sample_nums, dtype='float')
    dimensionality_lower_ci = np.zeros_like(shuff_cell_sample_nums)  # 5th percentile of bootstrapped dimensionality
    dimensionality_upper_ci = np.zeros_like(shuff_cell_sample_nums)  # 95th percentile of bootstrapped dimensionality

    # Transform data to z-score to center it as the units are not the same for all neurons
    Z = zscore(neurons, axis=1)
    Z = np.nan_to_num(Z)

    for i,cell_sample_num in tqdm(enumerate(shuff_cell_sample_nums), total=len(shuff_cell_sample_nums)):  

        # Create list of smaller arrays to pass to multiprocessing function
        array_subsets = []
        sample_func = sampling_func_lookup[sampling_method]
        for rep in range(pca_repetitions):
            temp_array = sample_func(Z, cell_sample_num, **kargs)
            array_subsets.append(temp_array)

        # Calculate dimensionality for all random samples
        dimensionality_bootstrap = []
        cutoff_array = np.ones(pca_repetitions)*cum_var_cutoff
        if use_multiprocessing:
            pool = Pool()
            for x in pool.starmap(_get_pca_dimensionality, zip(array_subsets, cutoff_array)):
                dimensionality_bootstrap.append(x)
            pool.close()
            pool.join()
        else:
            for array_subset in array_subsets:
                dimensionality = _get_pca_dimensionality(array_subset, cum_var_cutoff)
                dimensionality_bootstrap.append(dimensionality)

        # Save relevant values
        dimensionality_means[i] = np.mean(dimensionality_bootstrap)
        dimensionality_lower_ci[i] = np.percentile(dimensionality_bootstrap, 5)
        dimensionality_upper_ci[i] = np.percentile(dimensionality_bootstrap, 95)

    # Unshuffle arrays
    sorted_idx = np.argsort(shuff_cell_sample_nums)
    dimensionality_means = dimensionality_means[sorted_idx]
    dimensionality_lower_ci = dimensionality_lower_ci[sorted_idx]
    dimensionality_upper_ci = dimensionality_upper_ci[sorted_idx]

    return dimensionality_means, dimensionality_lower_ci, dimensionality_upper_ci


def _get_pca_dimensionality(array, cutoff):
    """ 
    Returns the dimensionality of the given array, defined as the number of PCA components 
    needed to exceed the cutoff of cumulative variance explained.
    """
    
    data_pca = PCA(n_components = array.shape[1]).fit(array)
    cum_var_explained = np.cumsum(data_pca.explained_variance_)/np.sum(data_pca.explained_variance_)
    dimensionality = np.where(cum_var_explained > cutoff)[0][0]
    return dimensionality

