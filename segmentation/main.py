import subprocess

import numpy as np

from info import optical_path_clean, outputs, otb
from segmentation.common import stack_layer

# stack layers
inputs = [im for im in optical_path_clean.glob('*.tif')]
stack_layer(inputs, 1, outputs / 'stack.tif', 100, np.int16)

# run otb Large scale segmentation
# Spatial radius of the neighborhood.
# Range radius defining the radius (expressed in radiometry unit) in the multispectral space.
range_r = [50, 70, 90, 100]
spatial_r = [20, 20, 20, 20]
for range, spatial in zip(range_r, spatial_r):
    subprocess.run(
        f'{otb}/otbcli_LargeScaleMeanShift.bat -in {outputs}/stack.tif -spatialr {spatial} -ranger {range} -mode raster -mode.raster.out {outputs}/seg_rad{range}_sp{spatial}.tif -ram 5000')
    subprocess.run(
        f'otbcli_LSMSVectorization -in {outputs}/stack.tif -inseg {outputs}/seg_rad{range}_sp{spatial}.tif -out {outputs}/seg_rad{range}_sp{spatial}.shp')
