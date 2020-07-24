import numpy as np
import seaborn as sns
from tqdm import tqdm
from scipy.stats import zscore
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from multiprocessing import Process, Manager, Pool

from Code import sampling
from Code.file_io import load_spontaneous,load_orientations

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
                                 z_transform_data=True, sampling_method='sample_uniform', 
                                 use_multiprocessing=False, n_component_guess_multiplier=0.5, **kargs):
    """ Return a curve of variance explained. Extra arguments are passed to the sampling function.
    
    Warnings: 1) Returned data will be sorted from lowest to highest cell_sample_nums.
    
    :param neurons: 2D array. Raw data. MUST be in the shape Timepoints x Neurons.
    :param cell_sample_nums: 1D Int array. Contains sample numbers to use.
    :param cum_var_cutoff: Float. Between 0 and 1. Cutoff for cumulative variance explained.
    :param pca_repetitions: Int. Number of PCA repeats for each sample_num
    :param z_transform_data: Bool. Set to True to z-score your array before processing
    :param sampling_method: Str. Unused at this time.
    :param use_multiprocessing: Bool. Set to False if multiprocessing functions throw errors.
    :param n_component_guess_multiplier: Float. Assume dimensionality ≈ [min(col_n, row_n)*this value]
    
    Returns three lists: dimensionality means, lower confidence intervals, and upper confidence intervals
    """

    sampling_func_lookup = {'sample_uniform': sampling.sample_uniform}
    
    if np.any(np.array(cell_sample_nums) > neurons.shape[1]):
        raise Exeception('Warning: More samples than neurons available requested!')
    
    # This is shuffled to better estimate runtime in TQDM
    shuff_cell_sample_nums = np.copy(cell_sample_nums)
    np.random.shuffle(shuff_cell_sample_nums)

    # Create empty arrays to store values
    dimensionality_means = np.zeros_like(shuff_cell_sample_nums, dtype='float')
    dimensionality_lower_ci = np.zeros_like(shuff_cell_sample_nums)  # 5th percentile of bootstrapped dimensionality
    dimensionality_upper_ci = np.zeros_like(shuff_cell_sample_nums)  # 95th percentile of bootstrapped dimensionality

    # Transform data to z-score to center it as the units are not the same for all neurons
    Z = neurons
    if z_transform_data:
        Z = zscore(Z, axis=0)
    Z = np.nan_to_num(Z)

    for i,cell_sample_num in tqdm(enumerate(shuff_cell_sample_nums), total=len(shuff_cell_sample_nums)):

        # Create list of smaller arrays to pass to multiprocessing function
        array_subsets = []
        sample_func = sampling_func_lookup[sampling_method]
        for rep in range(pca_repetitions):
            temp_array = sample_func(Z, cell_sample_num, **kargs)
            array_subsets.append(temp_array)

        # Calculate dimensionality for all random samples
        dimensionality_guess = np.ceil(cell_sample_num*n_component_guess_multiplier).astype('int')
        dimensionality_bootstrap = []
        if use_multiprocessing:
            cutoff_array = np.ones(pca_repetitions)*cum_var_cutoff
            dimensionality_guess_array = (np.ones(pca_repetitions)*dimensionality_guess).astype('int')
            pool = Pool()
            for x in pool.starmap(get_pca_dimensionality, zip(array_subsets, cutoff_array, dimensionality_guess_array)):
                dimensionality_bootstrap.append(x)
            pool.close()
            pool.join()
        else:
            for array_subset in array_subsets:
                dimensionality = get_pca_dimensionality(array_subset, cum_var_cutoff, dimensionality_guess)
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


def get_sample_cov_matrix(X):
    """
    Returns the sample covariance matrix of data X.

    :param 
    Args:
    X (numpy array of floats) : Data matrix each column corresponds to a
                                different random variable

    Returns:
    (numpy array of floats)   : Covariance matrix
    """

    X = X - np.mean(X, 0)
    cov_matrix = 1 / X.shape[0] * np.matmul(X.T, X)
    return cov_matrix


def get_pca_dimensionality(array, cutoff, n_components=None, covariance=None, z_transform_data=False, counter=0):
    """ 
    Returns the dimensionality of the given array, defined as the number of PCA components 
    needed to exceed the cutoff of cumulative variance explained.
    
    :param array: 2d numpy array. MUST be in Timepoints x Neurons shape.
    :param cutoff: Float. Dimensionality is assigned when cumulative variance explained > cutoff.
    :param n_components: Int. Number of components to calculate for to find PCA
    :param z_transform_data: Bool. Set to true if you want your data to be z-scored before processing
    
    """
    
    if n_components is None:
        n_components = np.min(array.shape)
    if z_transform_data:
        array = zscore(array, axis=0)
    if covariance is None:
        covariance = np.trace(get_sample_cov_matrix(array))
    data_pca = PCA(n_components = n_components).fit(array)
    cum_var_explained = np.cumsum(data_pca.explained_variance_)/covariance
    cum_var_thresholded = cum_var_explained > cutoff
    if np.sum(cum_var_thresholded) == 0:
        new_n_components = int(np.min((n_components+np.ceil(n_components*0.75), np.min(array.shape))))
        dimensionality = get_pca_dimensionality(array, cutoff, new_n_components, covariance, False ,counter+1)
    else:
        dimensionality = np.where(cum_var_thresholded)[0][0]
    return dimensionality

