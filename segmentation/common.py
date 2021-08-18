"""stack bands in one raster file"""
import rasterio


def stack_layer(input_images, selected_band, stack):
    """
    Args
        input_images (list): list contains Path for time series, images are supposed to have the same dimension
        selected_band (int): the band to use in the stack

    Return
        stack ( ): stack image
    """
    # get meta from a image
    with rasterio.open(input_images[0]) as src:
        meta = src.meta
    # open an image to save outputs
    meta.update({'band': len(input_images)})
    with rasterio.open(stack, 'w', **meta) as dst:
        # read src images, select the desired band and save it into the dst
        for im in input_images:
            with rasterio.open(im) as src:
                data = src.read(selected_band)
            dst.write(data, 1)
    return stack
