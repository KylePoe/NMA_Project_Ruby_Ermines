from Code.Tests.dim_analysis import dim_analysis
import numpy as np
from pathlib import Path
import timeit
from datetime import datetime

now = datetime.now()

analysis_folder = Path(f'dim_analysis_{now}')
analysis_folder.mkdir()

start = timeit.default_timer()

def time_update(section_name: str, value, section_start=None):
    time = timeit.default_timer()
    elapsed = time - start
    if section_start is None:
        update = f'\n ====={section_name} {value} started ({int(elapsed % 60)}m {int(elapsed / 60)}s since start)'
    else:
        section_time = time - section_start
        update = f'\n ====={section_name} {value} ended in {int(section_time % 60)}m {int(section_time/60)}s ({elapsed % 60}m {int(elapsed/60)}s since start)'

    print(update)

    with open(analysis_folder / 'log.txt', 'a') as fp:
        fp.write(update)

    return time

for bootstrap_window in (500*np.array(range(5, 20))):
    boot_folder = analysis_folder / f'{bootstrap_window} bootstrap window'
    boot_folder.mkdir()
    boot_start = time_update('Bootstrap window', bootstrap_window)

    for N_size in range(10, 21):
        n_folder = boot_folder / f'{N_size} neuron samples'
        n_folder.mkdir()
        n_start = time_update('Sample number', N_size)

        for trials_per_size in range(5, 21):
            trial_start = time_update('Trials per size', trials_per_size)
            params = {
                'bootstrap window': bootstrap_window,
                'trials_per_size': trials_per_size,
                'bootstrap_size': 30,
                'min_sample': 10,
                'max_sample': 11983,
                'N_sizes': N_size,
                'sampling_variance': 100,
                'uniform': False,
                'dataset': 'spontaneous'
            }

            dim_analysis(params, fname=n_folder / f'{trials_per_size} trials at {datetime.now()}')
            time_update('Trials per size', trials_per_size, trial_start)

        time_update('Sample number', N_size, n_start)

    time_update('Bootstrap window', bootstrap_window, boot_start)

print(f'\n\nTOTAL TIME TAKEN: {timeit.default_timer() - start}')





