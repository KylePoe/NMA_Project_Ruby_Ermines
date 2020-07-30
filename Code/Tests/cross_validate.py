import numpy as np
from Code import sampling as sa
from Code.visualization import scatterplot
from sklearn.decomposition import PCA
from scipy.stats import zscore
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import r2_score
import Code.file_io as load

mpl.use('Qt5Agg')
from Code.pca_variance import get_pca_dimensionality


def fit_to_test(X_train, X_test, n_components):
    model = PCA(n_components=n_components).fit(X_train)
    print(model.explained_variance_ratio_.sum())
    print(
        f"training data variance explained: {r2_score(X_train, model.inverse_transform(model.transform(X_train)), multioutput='variance_weighted')}")
    print(
        f"test data variance explained: {r2_score(X_test, model.inverse_transform(model.transform(X_test)), multioutput='variance_weighted')}")


def explained_variance(X, model):
    """ Returns the percent variance of X explained for each component of the provided PCA model"""

    result = np.zeros(model.n_components_)
    diff_mean_norm = np.linalg.norm(X - model.mean_)
    X_trans = model.transform(X)
    for ii in range(model.n_components_):
        X_trans_ii = np.zeros_like(X_trans)
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
        t: time points to sample for the train and test set
        control: Bool of whether or not you want to return control data

    Returns:
      training and testing data, plus an optional control data which is randomly shuffled training data.
    """

    if t is None or t > data.shape[0]:
        t = data.shape[0]//2

    # Pick a random selection of the time point range for the training range and the test range:
    ranges = np.random.choice(np.arange(data.shape[0]), size=t*2, replace=False)
    train_range = ranges[:t]
    test_range = ranges[t:]
    if control:
        return data[train_range], data[test_range], np.random.permutation(data[train_range].T).T
    else:
        return data[train_range], data[test_range]


def train_and_test_scree(X_train, X_test, model=None, n_components=None):
        """ Returns the percent cumulative variance explained as a function
        of number of components from the training data PCA model.

        Args:
            X_train: training data
            X_test: test data
            model: optional, PCA model of the test data
            n_components: optional, number of components for PCA model

        Returns:
          percent cumulative variance explained for training data, test data
           """

        if model is None:
            if n_components is None:
                model = PCA().fit(X_train)
            else:
                model = PCA(n_components=n_components).fit(X_train)
        train_variance_explained = np.cumsum(
            model.explained_variance_ratio_) * 100  # / np.sum(model.explained_variance_) * 100
        test_variance_explained = explained_variance(X_test, model)
        test_variance_explained = np.cumsum(test_variance_explained)
        x = np.arange(model.n_components_)
        return train_variance_explained, test_variance_explained


def scree(y, fig_title, labels):
    """
    Plots total variance explained versus number of components

    Args:
        y: (np.array of floats) number of y's to plot, shape n_components
        fig_title: (string) figure title
        labels: (list of strings): data labels

    Returns:
      Nothing.
    """
    x = np.arange(y.shape[1]) + 1
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(y.shape[0]):
        if i % 2 == 1:
            ls = '--'
        else:
            ls = '-'
        ax.plot(x, y[i, :], label=labels[i], ls=ls)
    ax.set_title(fig_title)
    ax.set_xlabel('N components')
    ax.set_ylabel('Variance explained (%)')
    ax.legend()
    plt.show()


def get_test_var(model, X_test):
    return r2_score(X_test, model.inverse_transform(model.transform(X_test)), multioutput='variance_weighted')


def test_vs_train_variance(neurons, repeats, t=None, n=None, n_set=None, t_set=None, variance_explained=0.8):
    """
    Compares the test and training set variance for the dimensions found with the training set.
    Can fix either the number of time points or the number of neurons.
    Neurons are randomly sampled uniformly.

    Args:
        neurons: time points by neurons (make sure it's z-scored!)
        t: time points to sample for training and test set
        n: number of neurons to sample
        repeats: number of repeats for each n in n_set, or t in t_set
        n_set: list of number of neurons to iterate over
        t_set: list of number of time points to iterate over (will be divided into two for train and test)
        variance_explained: desired variance explained for the test set

    Returns:
        np.array with columns:
        0: Number of time points
        1: Number of neurons
        2: Number of dimensions that describe the test set with variance_explained
        3: Test set cumulative variance explained for the number of dimensions in the PCA model
        4: Training set cumulative variance explained for the number of dimensions in the PCA model"""

    if n_set is not None:
        set = n_set
    else:
        set = t_set

    rows = len(set) * repeats
    out = np.empty((rows, 5)) #columns are number of time points, number of neurons, dimension, train_var, test_var

    k = 0
    for s in set:
        if n_set is not None:
            n = s
        else:
            t = s
        for _ in range(repeats):
            data = sa.sample_uniform(neurons, n=n)
            train, test = split_train_and_test(data, t=t)
            model = PCA(n_components=variance_explained).fit(train)
            out[k, 0] = t
            out[k, 1] = n
            out[k, 2] = model.n_components_
            out[k, 3] = np.cumsum(model.explained_variance_ratio_)[-1]
            out[k, 4] = get_test_var(model, test)
            k += 1
    return out


def get_test_dimension(neurons, t=None, n=None, n_set=None, t_set=None, repeats=3, threshold=0.8):
    """
    Compares the dimensions require to explain the threshold variance of the test set.

    Args:
        neurons: time points by neurons (make sure it's z-scored!)
        t: time points to sample for training and test set
        n: number of neurons to sample
        repeats: number of repeats for each n in n_set, or t in t_set
        n_set: list of number of neurons to iterate over
        t_set: list of number of time points to iterate over (will be divided into two for train and test)
        threshold: desired variance explained

    Returns:
        dict with indices:
        'n': Number of neurons
        't': Number of time points
        'train_dim': Number of dimensions that describe the training set with threshold variance
        'test_dim': Number of dimensions (from trained PCA model) that describe the test set with threshold variance
        'train_var': Training set cumulative variance explained for the number of dimensions in the PCA model
        'test_var': Test set cumulative variance explained for the number of dimensions in the PCA model"""
    if n_set is not None:
        set = n_set
    else:
        set = t_set

    rows = repeats*len(set)
    keys = ['n', 't', 'train_dim', 'test_dim', 'train_var', 'test_var']
    out = {key: np.empty(rows) for key in keys}
    i = 0
    for s in set:
        if n_set is not None:
            n = s
        else:
            t = s
        for _ in range(repeats):
            select_neurons = sa.sample_uniform(neurons, n)
            train, test = split_train_and_test(select_neurons, t=t)
            model = PCA().fit(train)
            total_train_var = np.cumsum(model.explained_variance_ratio_)
            train_dim = np.where(total_train_var > threshold)[0][0] + 1
            train_var = total_train_var[train_dim - 1]
            result = np.zeros(model.n_components_)
            diff_mean_norm = np.linalg.norm(test - model.mean_)
            test_trans = model.transform(test)
            for ii in range(train_dim-1, model.n_components_):
                test_trans_ii = np.zeros_like(test_trans)
                test_trans_ii[:, :ii+1] = test_trans[:, :ii+1]
                test_approx_ii = model.inverse_transform(test_trans_ii)
                result[ii] = 1 - (np.linalg.norm(test_approx_ii - test) /
                                      diff_mean_norm) ** 2
                if result[ii] > threshold:
                    test_dim = ii + 1
                    break
            if 'test_dim' not in locals():
                test_dim = 0
                print('test_dim was set to 0 because not enough components were fit.')
            out['n'][i] = n
            out['t'][i] = t
            out['train_dim'][i] = train_dim
            out['test_dim'][i] = test_dim
            out['train_var'][i] = train_var
            out['test_var'][i] = result[train_dim - 1]
            i += 1
    return out


def run_get_test_dimension(loading_function):

    dat = loading_function()
    neurons = dat['sresp'].T
    neurons = zscore(neurons, axis=0)
    all_t = neurons.shape[0]
    all_n = neurons.shape[1]

    # parameters for running over different numbers of time points:
    n = 10000 # to keep consistent between spontaneous and orientations
    t_set = [100, 200, 400, 800, 1600, all_t//4, all_t//2]

    # parameters for running over different numbers of neurons:
    t = 4598//2 # to keep consistent between spontaneous and orientations
    n_set = [100, 200, 400, 800, 1600, all_n//4, all_n//2, all_n]

    # run over time points:
    over_time = get_test_dimension(neurons, n=n, t_set=t_set)

    # run over neurons:
    over_neurons = get_test_dimension(neurons, t=t, n_set=n_set)

    return over_time, over_neurons



if __name__ == '__main__':
    # Run the following to sample random neurons from different depths and cross-validate the PCA model
    dat = load.load_orientations()
    neurons = dat['sresp'].T
    neuron_locs = dat['xyz']

    # Z score the data
    neurons = zscore(neurons, axis=0)

    # Set parameters
    n = 10  # number of neurons to sample
    t = 200  # number of time points to sample
    n_components = None  # number of components for PCA; if none it will be auto-set
    sup_range = (0, -300)  # superficial depth range; maximum of 12392 neurons for orientations data
    dep_range = (-301, -600)  # deep depth range; maximum of 11197 neurons for orientations data

    # select neurons in the depth range
    sup_data = sa.sample_depth_range(neurons, neuron_locs, sup_range, n=n)
    dep_data = sa.sample_depth_range(neurons, neuron_locs, dep_range, n=n)

    # split into training and test data
    sup_train, sup_test = split_train_and_test(sup_data, t=t)
    dep_train, dep_test = split_train_and_test(dep_data, t=t)

    # get cumulative variance explained
    sup_train_var, sup_test_var = train_and_test_scree(sup_train, sup_test, n_components=n_components)
    dep_train_var, dep_test_var = train_and_test_scree(dep_train, dep_test, n_components=n_components)

    # plot the data
    y = np.array([sup_train_var, sup_test_var, dep_train_var, dep_test_var])
    labels = ['Superficial training', 'Superficial test', 'Deep training', 'Deep test']
    scree(y, f'Orientations Data, {t} time points', labels)
    print(f'time points: {t}; neurons: {n}')
