from itertools import combinations

import pandas as pd
from numpy import trapz
from scipy.ndimage import gaussian_filter
from scipy.stats import linregress


def get_bi_combination(a_list):
    """gives a bi-combination for list items

    Args:
        a_list (list): list col names to apply combination of 2 elements

    Returns:
        combination
    """
    all_combinations = [i for i in combinations(a_list, 2)]
    return all_combinations


def peaks_pos(df, col_date, data):
    """ Gives the position (x-value) of the peak (max value)
    Args:
        df (Dataframe): data frame of data
        col_date (str): name of the column containing the x-data (here datetime)
        data (str): name of the column containing the y-data (SAR signal, NDVI, ....)
    Returns:
        list of peaks position
    """

    peaks = []
    for i, (a, b) in enumerate(zip(df[data], df[data][1:])):
        if b < a:
            peaks.append(df[col_date].iloc[i])
            break
    return peaks


def get_curve_length(df, col_date, data, date1, date2):
    """ estimate the curve length from  between two dates
    Args:
        df (Dataframe): data frame of data
        col_date (str): name of the column containing the x-data (here datetime)
        data (str): name of the column containing the y-data (SAR signal, NDVI, ....)
        date1(str): first date %Y-%M-%d
        date2(str):last date %Y-%M-%d
    Returns:
        list of peaks position
    """

    def pythagore(a, b):
        dist = (a ** 2 + b ** 2) ** 0.5
        return dist

    df = df.loc[df[col_date].between(date1, date2, inclusive='both')]
    df = df[[data, col_date]]
    df = df.diff()
    df = df.dropna()

    length = [pythagore(a, b) for a, b in zip(df[col_date].dt.days, df[data])]

    length = sum(length)

    return length


def gaussian_smoothing(df, col_data_to_smooth, sigma):
    """ Smooth a time series with a guassian function
    Args:
        df (DataFrame): dataframe of data
        col_data_to_smooth (str): nam of columns to smooth
        sigma (float): smooth level
    Return:
        df (DataFrame): the input dataframe with smoothed data
    """

    df[f'{col_data_to_smooth}_smooth'] = gaussian_filter(df[col_data_to_smooth], sigma)

    return df


def slope_r2(x, y):
    """ get slope of temporal series
    Args:
        x (Datetime): datetime
        y (list): variable
    Return:
        slope (float): slope
    """
    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    return slope, r_value ** 2


def get_area_under_curve(df, col_date, data, date1, date2):
    """ get the area under a curve
    Args:
        df (Dataframe): data frame of data
        col_date (str): name of the column containing the x-data (here datetime)
        data (str): name of the column containing the y-data (SAR signal, NDVI, ....)
        date1(str): first date %Y-%M-%d
        date2(str):last date %Y-%M-%d
    Returns:
        (float) area under the curve
    """
    df = df.loc[df[col_date].between(date1, date2, inclusive='both')]

    area = trapz(df[data], x=(df[col_date] - df[col_date].min()).dt.days)

    return area


def slope_moving_windows(df, windows, col_date, col):
    """ gives the slope of the time series
    Args:
        df (Dataframe): data frame of data
        windows (int): windows length 3,5,7,.....
        col_date (str): name of the column containing the x-data (here datetime in days)
        col_date (str): name of the column containing the y-data (NDVI, SAR scattering)

    Returns:
        (list): list containing slopes
    """
    df_windows = [df.iloc[i:i + windows] for i in range(len(df[col]) - windows + 1)]
    slope = [slope_r2((sdf[col_date] - sdf[col_date].min()).dt.days, sdf[col])[0] for sdf in df_windows]

    return slope


def generate_features(df, plot_name, plot_class1, plot_class2, cols_predictive, col_date, clos_cmp=None):
    """Generates features (input to RF classifier)
    Args:
        df (DataFrame): stats dataframe wih index date
        plot_name (str): name of cols of plots name
        plot_class1 (str): col name for type of the plot (1st half of the year)
        plot_class2 (str): col name for type of the plot (2nd half of the year)
        cols_predictive (list): list of columns name of predictive variables
        col_date (str): name of the cols contains the date information
        clos_cmp (list): list of cols to compare data
    Return:
        df (DataFrame): dataframe of features
    """

    # get features
    all_features = []

    # sort values by plot name and dates
    df = df.sort_values([plot_name, col_date])

    for nb, sdf in df.groupby(by=plot_name):
        features = {}
        features.update({f'label': nb})
        features.update({f'ref_class1': sdf[plot_class1].unique()[0]})
        features.update({f'ref_class2': sdf[plot_class2].unique()[0]})

        # get feature variable from each predictive col
        for col in cols_predictive:

            # compute the area of the curve
            area = get_area_under_curve(sdf, col_date, col, '2015-1-1', '2015-12-31')
            features.update({f'area_{col}': area})

            # get moving slope
            slopes = slope_moving_windows(sdf, 3, col_date, col)
            for i, s in enumerate(slopes):
                features.update({f'slope_{i}_{col}': s})
            # two cols corr
            if clos_cmp:
                # get combination with size of 2
                comb = get_bi_combination(clos_cmp)
                for c in comb:
                    el1 = c[0]
                    el2 = c[1]
                    # correlation between two elements
                    _, r = slope_r2(sdf[el1], sdf[el2])
                    features.update({f'r2_{el1}_{el2}': r})

        all_features.append(features)
    df_features_train = pd.DataFrame(all_features)

    return df_features_train
