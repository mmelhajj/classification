import geopandas as gpd
import numpy as np


def get_data_from_shape(shape, cols=None, save_csv=None, path_csv=None, name_csv=None):
    """ converts shape to csv
    Args:
        shape (Path): path and name of the shape file
        cols (list): list of columns to keep
        save_csv (bool): True save, False don't save
        path_csv (Path): Path to save csv
        name_csv (str): csv name
   Return:
        gdf (DataFrame): dataframe of data
    """
    gdf = gpd.read_file(shape)
    if cols:
        gdf = gdf[cols]

    # replace None of shape with np.nan
    gdf = gdf.fillna(value=np.nan)

    if save_csv:
        with open(path_csv / name_csv, 'w') as fmt:
            gdf.to_csv(fmt, index=False, line_terminator='\n')

        return gdf, fmt

    return gdf
