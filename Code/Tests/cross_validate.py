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
        t: total time points to sample (will be divided into two for train and test)
        control: Bool of whether or not you want to return control data

    Returns:
      training and testing data, plus an optional control data which is randomly shuffled training data.
    """

    if t is None or t > data.shape[0]:
        t = data.shape[0]

    # Pick a random selection of the time point range for the training range and the test range:
    ranges = np.random.choice(np.arange(data.shape[0]), size=t, replace=False)
    train_range = ranges[:t//2]
    test_range = ranges[t//2:]
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

if __name__ == '__main__':
    # Run the following to sample random neurons from different depths and cross-validate the PCA model
    dat = load.load_orientations()
    neurons = dat['sresp'].T
    neuron_locs = dat['xyz']

    # Z score the data
    neurons = zscore(neurons, axis=0)

    # Set parameters
    n = 10  # number of neurons to sample
    t = 200  # number of time points to sample (will be divided into 2 for training and test)
    n_components = None  # number of components for PCA; if none it will be auto-set
    sup_range = (0, -300)  # superficial depth range; maximum of 12392 neurons for orientations data
    dep_range = (-301, -600)  # deep depth range; maximum of 11197 neurons for orientations data

    # select neurons in the depth range
    sup_data = spl.sample_depth_range(neurons, neuron_locs, sup_range, n=n)
    dep_data = spl.sample_depth_range(neurons, neuron_locs, dep_range, n=n)

    # split into training and test data
    sup_train, sup_test = split_train_and_test(sup_data, t=t)
    dep_train, dep_test = split_train_and_test(dep_data, t=t)

    # get cumulative variance explained
    sup_train_var, sup_test_var = train_and_test_scree(sup_train, sup_test, n_components=n_components)
    dep_train_var, dep_test_var = train_and_test_scree(dep_train, dep_test, n_components=n_components)

    # plot the data
    y = np.array([sup_train_var, sup_test_var, dep_train_var, dep_test_var])
    labels = ['Superficial training', 'Superficial test', 'Deep training', 'Deep test']
    scree(y, f'Orientations Data, {t // 2} time points', labels)
    print(f'time points: {t}; neurons: {n}')
