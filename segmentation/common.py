"""stack bands in one raster file"""
import rasterio


def stack_layer(input_images, selected_band, stack, scale, output_type=None):
    """
    Args
        input_images (list): list contains Path for time series, images are supposed to have the same dimension
        selected_band (int): the band to use in the stack
        scale (int): scale used
        output_type (type): data output type after scaling
    Return
        stack (DatasetWriter): stack image
    """
    # get meta from a image
    with rasterio.open(input_images[0]) as src:
        meta = src.meta
    # open an image to save outputs
    meta.update({'count': len(input_images)})
    if output_type:
        meta.update({'dtype': output_type})

    with rasterio.open(stack, 'w', **meta) as dst:
        # read src images, select the desired band and save it into the dst
        for b, im in enumerate(input_images):
            with rasterio.open(im) as src:
                data = src.read(selected_band) * scale

            # write image
            dst.write(data.astype(output_type), b + 1)
    return stack
