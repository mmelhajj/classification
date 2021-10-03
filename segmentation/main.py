import subprocess

import numpy as np

from info import outputs, otb, L8_path_clean
from segmentation.common import stack_layer

# stack layers
inputs = [im for im in L8_path_clean.glob('*.tif')]
outputs = outputs / 'L8'
output_name = 'stack_L8_ndvi.tif'
stack_layer(inputs, 1, outputs / output_name, 100, np.int16)

# run otb Large scale segmentation
# Spatial radius of the neighborhood.
# Range radius defining the radius (expressed in radiometry unit) in the multispectral space.
range_r = [50, 70, 90, 100]
spatial_r = [20, 20, 20, 20]
for range, spatial in zip(range_r, spatial_r):
    subprocess.run(
        f'{otb}/otbcli_LargeScaleMeanShift.bat -in {outputs}/{output_name} -spatialr {spatial} -ranger {range} -mode raster -mode.raster.out {outputs}/seg_rad{range}_sp{spatial}.tif -ram 5000')
    subprocess.run(
        f'{otb}/otbcli_LSMSVectorization.bat -in {outputs}/{output_name} -inseg {outputs}/seg_rad{range}_sp{spatial}.tif -out {outputs}/seg_rad{range}_sp{spatial}_L8.shp')
