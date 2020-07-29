import json
import hashlib
import numpy as np
from os import path
import seaborn as sns
from tqdm import tqdm
from scipy.stats import zscore
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
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
                                 use_multiprocessing=False, return_dict=False, depth_range=None, 
                                 neuron_locs=None, **kwargs):
    """ Return a curve of variance explained. Extra arguments are passed to the sampling function.
    
    Warnings: 1) Returned data will be sorted from lowest to highest cell_sample_nums.
    
    :param neurons: 2D array. Raw data. MUST be in the shape Timepoints x Neurons.
    :param cell_sample_nums: 1D Int array. Contains sample numbers to use.
    :param cum_var_cutoff: Float. Between 0 and 1. Cutoff for cumulative variance explained.
    :param pca_repetitions: Int. Number of PCA repeats for each sample_num
    :param z_transform_data: Bool. Set to True to z-score your array before processing
    :param sampling_method: Str. Unused at this time.
    :param use_multiprocessing: Bool. Set to False if multiprocessing functions throw errors.
    
    Returns three lists: dimensionality means, lower confidence intervals, and upper confidence intervals
    """

    sampling_func_lookup = {'sample_uniform': sampling.sample_uniform,
                            'sample_around_point': sampling.sample_around_point,
                            'sample_depths_uniform': sampling.sample_depth_range,
                            'sample_depths_point': sampling.sample_around_point}
    sample_func = sampling_func_lookup[sampling_method]
    
    if np.any(np.array(cell_sample_nums) > neurons.shape[1]):
        raise Exception('Warning: More samples than neurons available requested!')
    
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
    
    # Filter dataset to only include depth range if sample_depths_point used
    if sampling_method == 'sample_depths_point':
        upper,lower = (np.max(depth_range), np.min(depth_range))
        mask = np.where(np.logical_and(neuron_locs[2,:] <= upper, neuron_locs[2,:] >= lower))[0]
        Z = Z[:, mask]
        neuron_locs = np.array(neuron_locs)[:,mask]
    
    # Determine curve for dimensionality guess
    dim_sample_nums = [1000, 2000, 3000]
    dim_sample_results = []
    for dim_sample_num in dim_sample_nums:
        sample_neurons = sampling_func_lookup['sample_uniform'](neurons=Z, n=dim_sample_num, depth_range=depth_range, **kwargs)
        guess_dimensionality = int(np.min(sample_neurons.shape)*0.75)
        dim_sample_results.append(get_pca_dimensionality(sample_neurons, cum_var_cutoff, guess_dimensionality))
    dim_curve_params, _ = curve_fit(_dim_curve, dim_sample_nums, dim_sample_results, p0=(1, 1, 4000), maxfev=10000)

    full_data_dict = {}
    full_data_dict['neuron_nums'] = {}
    for i,cell_sample_num in tqdm(enumerate(shuff_cell_sample_nums), total=len(shuff_cell_sample_nums)):

        # Create list of smaller arrays to pass to multiprocessing function
        array_subsets = []
        for rep in range(pca_repetitions):
            temp_array = sample_func(Z, n=cell_sample_num, neuron_locs=neuron_locs, depth_range=depth_range, **kwargs)
            array_subsets.append(temp_array)

        # Calculate dimensionality for all random samples
        dimensionality_guess = int(np.min((_dim_curve(cell_sample_num, *dim_curve_params)+300, *array_subsets[0].shape)))
        dimensionality_bootstrap = []
        if use_multiprocessing:
            cutoff_array = np.ones(pca_repetitions)*cum_var_cutoff
            dimensionality_guess_array = (np.ones(pca_repetitions)*dimensionality_guess).astype('int')
            pool = Pool()
            for x in pool.starmap(get_pca_dimensionality, zip(array_subsets, cutoff_array, dimensionality_guess_array)):
                dimensionality_bootstrap.append(x)
            pool.close()
        else:
            for array_subset in array_subsets:
                dimensionality_bootstrap.append(dimensionality)

        # Save relevant values
        dimensionality_means[i] = np.mean(dimensionality_bootstrap)
        dimensionality_lower_ci[i] = np.percentile(dimensionality_bootstrap, 5)
        dimensionality_upper_ci[i] = np.percentile(dimensionality_bootstrap, 95)
        if return_dict:
            full_data_dict[str(cell_sample_num)] = dimensionality_bootstrap
            true_num_sampled = [t.shape[1] for t in array_subsets]
            if len(np.unique(true_num_sampled)) > 1:
                raise Exception(f'Warning: Number of neurons sampled is not consistent! Results: {true_num_sampled}')
            full_data_dict['neuron_nums'][str(cell_sample_num)] = np.mean(true_num_sampled)

    # Unshuffle arrays
    sorted_idx = np.argsort(shuff_cell_sample_nums)
    dimensionality_means = dimensionality_means[sorted_idx]
    dimensionality_lower_ci = dimensionality_lower_ci[sorted_idx]
    dimensionality_upper_ci = dimensionality_upper_ci[sorted_idx]

    if return_dict:
        return full_data_dict
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
        dimensionality = np.where(cum_var_thresholded)[0][0]+1
    return int(dimensionality)


def save_dim_data(params, data_dict):

    data_md5 = hashlib.md5(json.dumps(params, sort_keys=True).encode('utf-8')).hexdigest()
    filename = f'Data/{data_md5}.json'
    if path.exists(filename):
        #print('Param datafile found! Adding data...')
        with open(filename, 'r+') as jf:
            old_data_dict = json.load(jf)
            for key in data_dict.keys():
                if key in old_data_dict:
                    old_data_dict[key] = old_data_dict[key] + list(data_dict[key])
                else:
                    old_data_dict[key] = list(data_dict[key])
        jf.close()
        with open(filename, 'w') as jf:
            json.dump(old_data_dict, jf, sort_keys=True, indent=4)
    else:
        #print('Params datafile not found, creating new file...')
        with open(filename, 'w') as jf:
            data_dict['params'] = params
            json.dump(data_dict, jf, sort_keys=True, indent=4)
    return


def fetch_dim_data(params):
    
    data_md5 = hashlib.md5(json.dumps(params, sort_keys=True).encode('utf-8')).hexdigest()
    filename = f'Data/{data_md5}.json'
    if path.exists(filename):
        with open(filename, 'r') as jf:
            data_dict = json.load(jf)
    else:
        raise Exception(f'Error: File not found for given parameters.')
    return data_dict


def _dim_curve(data,a,b,c):
    return (a-b)*np.exp(-data/c)+b