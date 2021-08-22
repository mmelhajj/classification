import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


def get_data_from_shape(shape, cols, save_csv=False, path_csv=None, name_csv=None):
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
    gdf = gdf[cols]

    # replace None of shape with np.nan
    gdf = gdf.fillna(value=np.nan)

    if save_csv:
        with open(path_csv / name_csv, 'w') as fmt:
            gdf.to_csv(fmt, index=False, line_terminator='\n')

        return gdf, fmt

    return gdf


def set_training(stat_csv, label_csv, stat_on, label_on, how='inner', save_csv=False, path_csv=None, name_csv=None):
    """ create a csv that attribute to stats a label name, it merges stats csv and label csv
    Args:
        stat_csv (Dataframe): path and name of the stat csv
        label_csv (Path): path and name of the label csv
        stat_on (str): index of stat_csv to merge
        label_on (str): index of label_csv to merge
        how (str): merge type : how{‘left’, ‘right’, ‘outer’, ‘inner’, ‘cross’}, default ‘inner’
        save_csv (bool): True save, False don't save
        path_csv (Path): Path to save csv
        name_csv (str): csv name
   Return:
        stat_csv (DataFrame): dataframe of data
    """
    stat_csv = stat_csv.merge(label_csv, left_on=stat_on, right_on=label_on, how=how)
    if save_csv:
        with open(path_csv / name_csv, 'w') as fmt:
            stat_csv.to_csv(fmt, index=False, line_terminator='\n')
        return stat_csv, fmt

    return stat_csv


def gaussian_smoothing(df, col_data_to_smooth, sigma):
    """
    Args:
        df (DataFrame): dataframe of data
        col_data_to_smooth (str): name of columns to smooth
        sigma (float): smooth level
    Return:
        df (DataFrame): the input dataframe with smoothed data
    """

    df[f'{col_data_to_smooth}_smooth'] = gaussian_filter(df[col_data_to_smooth], sigma)

    return df


def smooth_variables(df, col_group_by, cols, sigma):
    """ Add more raw variables to zonal stats
    Args:
     df (Datafarme): data frame of raw data
     col_group_by (list): list of name of cols to isolate sample when adding more raw variables
     cols (list): name of cols to smooth
     sigma (int): order of smooth as sigma of the Gaussian
    """
    all_df = []
    for _, sdf in df.groupby(by=col_group_by):
        # apply smoothing only
        for cl in cols:
            sdf = gaussian_smoothing(sdf, cl, sigma)
        all_df.append(sdf)
    df = pd.concat(all_df)
    return df


def normalise_cols(df, cols):
    """ normalize data between 0 and 1
    Args:
        df (DataFrame): dataframe of data
        cols (list): name of columns to normalize
    Return:
        df (DataFrame): the input dataframe with smoothed data
    """
    for cl in cols:
        df[cl] = (df[cl] - df[cl].min()) / (df[cl].max() - df[cl].min())

    return df
