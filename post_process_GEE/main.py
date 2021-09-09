from info import sar_clean_path, sar_raw_path
from post_process_GEE.convert_to_L import convert_gee_s1_to_linear_scale

images = sar_raw_path.glob('*.tif')
for im in images:
    output_image = sar_clean_path / f"{im.name}"
    convert_gee_s1_to_linear_scale(im, output_image, [3, 5], 10000)
