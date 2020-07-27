import numpy as np
from Code import sampling as spl
from Code.visualization import scatterplot
from sklearn.decomposition import PCA
from scipy.stats import zscore
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import r2_score
import Code.file_io as load
mpl.use('Qt5Agg')
from Code.pca_variance import get_pca_dimensionality


def explained_variance(X, model):
    """ Returns the percent variance of X explained for each component of the provided PCA model"""

    result = np.zeros(model.n_components_)
    diff_mean_norm = np.linalg.norm(X - model.mean_)
    X_trans = model.transform(X)
    X_zeros = np.zeros_like(X_trans)
    for ii in range(model.n_components_):
        X_trans_ii = X_zeros
        X_trans_ii[:, ii] = X_trans[:, ii]
        X_approx_ii = model.inverse_transform(X_trans_ii)
        result[ii] = 1 - (np.linalg.norm(X_approx_ii - X) /
                          diff_mean_norm) ** 2
    return result * 100


def split_train_and_test(data, t=None, control=False):

    """
    Splits data into test and training

    Args:
        data: samples by parameters
        t: total time points to sample (will be divided into two for train and test)
        control: Bool of whether or not you want to return control data

    Returns:
      training and testing data, plus an optional control data which is randomly shuffled training data.
    """

    if t is None:
        t = data.shape[0]

    #Pick a random half of the time point range for the training range and the test range:
    range1 = np.arange(t//2)
    range2 = range1 + t//2
    ranges = np.array([range1, range2])
    train_range, test_range = np.random.permutation(ranges)

    if control:
        return data[train_range], data[test_range], np.random.permutation(data[train_range].T).T
    else:
        return data[train_range], data[test_range]


def train_and_test_scree(X_train=None, X_test=None, data=None, t=None, model=None, n_components=None, do_zscore=False):
    """ Returns the percent cumulative variance explained as a function
    of number of components from the training data PCA model.

    Args:
        X_train: training data (optional if provided data)
        X_test: test data (optional if provided data)
        data: samples by parameters, data to split into training and test (optional if provided X_train and X_test)
        t: optional, total number of time points to sample (will be divided into 2 for test and training)
        model: optional, PCA model of the test data
        n_components: optional, number of components for PCA model
        do_zscore: Bool of whether or not you want to do a zscore

    Returns:
      training and testing data, plus an optional control data which is randomly shuffled training data.
       """

    if X_train and X_test and data is None:
        raise Exception('You need to provide some sort of data.')
    elif data is not None:
        X_train, X_test = split_train_and_test(data, t)
    if do_zscore:
        X_train = zscore(X_train, axis=0)
        X_test = zscore(X_test, axis=0)
    if model is None:
        if n_components is None:
            model = PCA().fit(X_train)
        else:
            model = PCA(n_components=n_components).fit(X_train)
    train_variance_explained = np.cumsum(model.explained_variance_ratio_) * 100# / np.sum(model.explained_variance_) * 100
    test_variance_explained = explained_variance(X_test, model)
    test_variance_explained = np.cumsum(test_variance_explained)
    x = np.arange(model.n_components_)
    return train_variance_explained, test_variance_explained

