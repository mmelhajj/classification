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


def get_min_max_pos(df, col_date, data, date1, date2):
    """ Gives the position (x-value) of the min and the max value
    Args:
        df (Dataframe): data frame of data
        col_date (str): name of the column containing the x-data (here datetime)
        data (str): name of the column containing the y-data (SAR signal, NDVI, ....)
        date1(str): first date %Y-%M-%d
        date2(str):last date %Y-%M-%d
    Returns:
        list of peaks position
    """
    df = df.loc[df[col_date].between(date1, date2, inclusive='both')]

    date_of_min = pd.to_datetime(df.loc[df[data] == df[data].min()][col_date].values[0])
    day_of_min = (date_of_min - df[col_date].min()).days

    date_of_max = pd.to_datetime(df.loc[df[data] == df[data].max()][col_date].values[0])
    day_of_max = (date_of_min - df[col_date].max()).days

    return date_of_min, day_of_min, date_of_max, day_of_max


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

            # get first season start # TODO: use only NDVI
            date_of_min, doy_of_min, _, _ = get_min_max_pos(sdf, col_date, col, '2014-11-01', '2014-12-31')
            features.update({f'start_{col}': doy_of_min})

            # get first season peak: TODO: use only NDVI
            _, _, date_of_max, doy_of_max = get_min_max_pos(sdf, col_date, col, date_of_min.strftime('%Y-%m-%d'),
                                                            '2015-6-01')
            features.update({f'end_{col}': doy_of_max})

            # get the length of the curve from season start to middle of season
            length = get_curve_length(df, col_date, col, date_of_min.strftime('%Y-%m-%d'),
                                      date_of_max.strftime('%Y-%m-%d'))
            features.update({f'length_{col}': length})

            # compute the area of the curve
            area = get_area_under_curve(sdf, col_date, col, date_of_min.strftime('%Y-%m-%d'), '2015-12-31')
            features.update({f'area_{col}': area})

            # get the position of the max
            pos = pd.to_datetime(sdf.loc[sdf[col] == sdf[col].max()][col_date].values[0]).dayofyear
            features.update({f'pos_{col}': pos})

            # two cols corr
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
