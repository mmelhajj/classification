import pandas as pd
from numpy import trapz
from scipy.ndimage import gaussian_filter
from scipy.stats import linregress


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

    return slope


def generate_features(df, plot_nb, plot_class, cols_predictive, col_date):
    """Generates features (input to RF classifier)
    Args:
        df (DataFrame): stats dataframe wih index date
        plot_nb (str): name of cols of plots name
        plot_class (str): col name for type of the plot
        cols_predictive (list): list of columns name of predictive variables
        col_date (str): name of the cols contains the date information
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
            data_slope = sdf.loc[df[col_date].between('2020-01-1', '2020-7-1', inclusive='both')]
            slope = get_slope_from_temporal_series(data_slope[col_date].dt.strftime('%y%j').astype(float),
                                                   data_slope[col])

            features.update({f'slp_start_{col}': slope})

            # compute slope at end of the year
            data_slope = sdf.loc[df[col_date].between('2020-08-1', '2020-12-31', inclusive='both')]
            slope = get_slope_from_temporal_series(data_slope[col_date].dt.strftime('%y%j').astype(float),
                                                   data_slope[col])

            features.update({f'slp_end_{col}': slope})

            # compute slope for the whole year
            slope = get_slope_from_temporal_series(sdf[col_date].dt.strftime('%y%j').astype(float), sdf[col])

            features.update({f'slp_all_{col}': slope})

            # compute global variation
            features.update({f'var_{col}': sdf[col].var()})

            # compute the area of the curve
            area = trapz(sdf[col], dx=5)

            features.update({f'area_{col}': area})

        all_features.append(features)
    df_features_train = pd.DataFrame(all_features)

    return df_features_train
