from itertools import combinations

import pandas as pd
from numpy import trapz
from scipy.ndimage import gaussian_filter
from scipy.stats import linregress


def get_bi_combination(a_list):
    all_combinations = [i for i in combinations(a_list, 2)]
    return all_combinations


def gaussian_smoothing(df, col_data_to_smooth, sigma):
    """
    Args:
        df (DataFrame): dataframe of data
        col_data_to_smooth (str): nam of columns to smooth
        sigma (float): smooth level
    Return:
        df (DataFrame): the input dataframe with smoothed data
    """

    df[f'{col_data_to_smooth}_smooth'] = gaussian_filter(df[col_data_to_smooth], sigma)

    return df


def get_slope_from_temporal_series(x, y):
    """ get slope of temporal series
    Args:
        x (Datetime): datetime
        y (float): variable
    Return:
        slope (float): slope
    """
    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    return slope, r_value ** 2


def generate_features(df, plot_nb, plot_class, cols_predictive, col_date, clos_cmp=None):
    """Generates features (input to RF classifier)
    Args:
        df (DataFrame): stats dataframe wih index date
        plot_nb (str): name of cols of plots name
        plot_class (str): col name for type of the plot
        cols_predictive (list): list of columns name of predictive variables
        col_date (str): name of the cols contains the date information
        clos_cmp (list): list of cols to compare data
    Return:
        df (DataFrame): dataframe of features
    """

    # get features
    all_features = []

    for nb, sdf in df.groupby(by=plot_nb):
        features = {}
        features.update({f'label': nb})
        features.update({f'ref_class': sdf[plot_class].unique()[0]})

        # get feature variable from each predictive col
        for col in cols_predictive:
            # compute slope at start of the year
            data_slope = sdf.loc[df[col_date].between('2015-01-1', '2015-7-1', inclusive='both')]
            slope, _ = get_slope_from_temporal_series(data_slope[col_date].dt.strftime('%y%j').astype(float),
                                                      data_slope[col])
            features.update({f'slp_start_{col}': slope})

            # compute slope at end of the year
            data_slope = sdf.loc[df[col_date].between('2015-08-1', '2015-12-31', inclusive='both')]
            slope, _ = get_slope_from_temporal_series(data_slope[col_date].dt.strftime('%y%j').astype(float),
                                                      data_slope[col])
            features.update({f'slp_end_{col}': slope})

            # compute slope for the whole year
            slope, _ = get_slope_from_temporal_series(sdf[col_date].dt.strftime('%y%j').astype(float), sdf[col])
            features.update({f'slp_all_{col}': slope})

            # compute global variation
            features.update({f'var_{col}': sdf[col].var()})

            # compute annual mean
            data_mean = sdf.loc[df[col_date].between('2015-03-1', '2015-08-1', inclusive='both')]
            features.update(
                {f'mean_{col}': data_mean[col].mean()})

            # compute the area of the curve
            area = trapz(sdf[col], x=sdf[col_date].dt.strftime('%y%j').astype(float))
            features.update({f'area_{col}': area})

            if clos_cmp:
                # get combination with size of 2
                comb = get_bi_combination(clos_cmp)
                for c in comb:
                    el1 = c[0]
                    el2 = c[1]
                    # correlation between two elements
                    _, r = get_slope_from_temporal_series(sdf[el1], sdf[el2])
                    features.update({f'r2_{el1}_{el2}': r})

        all_features.append(features)
    df_features_train = pd.DataFrame(all_features)

    return df_features_train
