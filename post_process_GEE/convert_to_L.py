from pathlib import Path

import rasterio
import rasterio.mask

from info import sar_clean_path


def convert_gee_s1_to_linear_scale(input_image, output_image, exclude_bands, scale):
    """
    Args:
        input_image (Path): path of GEE image in dB
        output_image (Path): path of GEE image in Linear unit
        exclude_bands (list): list of bands not to convert to L
        scale (int): scale applied to input image

    Returns:
        dest (DatasetWriter): image in Linear
    """
    with rasterio.open(input_image) as src:
        # open to save output image
        out_meta = src.meta
        # update metadata
        out_meta.update({"dtype": 'float32'})
        with rasterio.open(output_image, "w", **out_meta) as destination:
            bands = src.count
            for band in range(bands):
                band += 1
                array = src.read(band).astype(float) / scale
                if band not in exclude_bands:
                    # convert to L
                    array = 10 ** (array / 10)

                destination.write(array.astype(rasterio.float32), band)

        return destination


