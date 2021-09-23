from pathlib import Path

import rasterio
import rasterio.mask

from info import L8_path_clean, L8_path_raw


def ndvi(input_image, red_band, nir_band, output_image):
    """
    Args:
        input_image (Path): path of GEE image
        red_band (path): image red band
        nir_band (path): image nir band
        output_image (str): path of ndvi output

    Returns:
        dest (DatasetWriter): image in Linear
    """
    with rasterio.open(input_image) as src:
        # open to save output image
        out_meta = src.meta
        # update metadata
        out_meta.update({"dtype": 'float64',
                         "count": '1'})
        # nir and red bands
        nir = src.read(nir_band).astype(float)
        red = src.read(red_band).astype(float)
        ndvi = (nir - red) / (nir + red)

        with rasterio.open(output_image, "w", **out_meta) as destination:
            destination.write(ndvi, 1)

        return destination


if __name__ == '__main__':
    images = L8_path_raw.glob('*.tif')
    for im in images:
        output_image = L8_path_clean/im.name
        ndvi(im, 1, 2, output_image)
