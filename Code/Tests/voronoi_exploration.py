from Code.sampling import voronoi_tesselation_3d, sample_uniform
import numpy as np
from Code.visualization import visualize_voronoi_tesselation

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

dat = np.load(data['Spontaneous']['file'], allow_pickle=True).item()

# Generate 20 regions for voronoi diagram
points = sample_uniform(dat['xyz'], n=5)
tesselation = voronoi_tesselation_3d(dat['xyz'], points)
visualize_voronoi_tesselation(tesselation)

