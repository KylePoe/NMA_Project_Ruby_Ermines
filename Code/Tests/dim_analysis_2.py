import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

data = np.load('/tmp/pycharm_project_70/dim_analysis_2020-07-28 04:13:30.285212/500 bootstrap window/5 neuron samples/5 trials at 2020-07-28 04:13:30.285825.npy', allow_pickle=True).item()
params = data['parameters']
mat = data['output']

fig = plt.figure()
ax = fig.add_subplot(111)

means = np.zeros((params['N_sizes'], params['trials_per_size']))
stds = np.zeros((params['N_sizes'], params['trials_per_size']))

for i, sample_size in enumerate(data['sample sizes']):

    means[i, :] = mat[i, :, :].mean(axis=1)
    stds[i, :] = mat[i, :, :].std(axis=1)

    ax.scatter(sample_size * np.ones_like(means[i, :]), means[i, :], c='r', marker='o')

ax.plot(data['sample sizes'], means.mean(axis=1), c='b', linestyle='-')

# Obtain fit

data_points = means.reshape(1, params['N_sizes'] * params['trials_per_size'])
neurons = np.kron(data['sample sizes'], np.ones(params['trials_per_size']))

def _dim_curve(data,a,b,c):
    return (a-b)*np.exp(-data/c)+b

dim_curve_params, pcov = curve_fit(_dim_curve, neurons, data_points.squeeze(), p0=(1, 1, 4000), maxfev=10000)

nn = np.linspace(neurons[0], neurons[-1])

ax.plot(nn, _dim_curve(nn, *dim_curve_params), c='g')

plt.show()
