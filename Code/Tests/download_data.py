#!/usr/env/bin python

# From https://github.com/NeuromatchAcademy/course-content/blob/master/projects/load_stringer_spontaneous.ipynb
import os, requests
import numpy as np

data = {
    'Orientation': {
        'url': "https://osf.io/ny4ut/download",
        'file': "Data/stringer_orientations.npy"
    },
    'Spontaneous': {
        'url': "https://osf.io/dpqaj/download",
        'file': "Data/stringer_spontaneous.npy"
    }
}

def download_data():
    for dataset in data.keys():
        if not os.path.isfile(data[dataset]['file']):
            try:
                print(f'Downloading {str(dataset)} data...')
                r = requests.get(data[dataset]['url'])
            except requests.ConnectionError:
                print("!!! Failed to download data !!!")

            else:
                if r.status_code != requests.codes.ok:
                    print("!!! Failed to download data !!!")

                else:
                    with open(data[dataset]['file'], "wb") as fid: 
                        fid.write(r.content)

if __name__ == '__main__':
    download_data()