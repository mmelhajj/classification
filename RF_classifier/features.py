import pandas as pd
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


def generate_features(df, plot_nb, plot_class, cols_predictive, col_date, for_train=True):
    """Generates features (input to RF classifier)
    Args:
        df (DataFrame): stats dataframe wih index date
        plot_nb (str): name of cols of plots name
        plot_class (str): col name for type of the plot
        cols_predictive (list): list of columns name of predictive variables
        col_date (str): name of the cols contains the date information
        for_train (bool)= True features for training, False feature for estimation
    Return:
        df (DataFrame): dataframe of features
    """

    # get features
    all_features = []

    for nb, sdf in df.groupby(by=plot_nb):
        features = {}
        features.update({f'label': nb})

        # get feature variable from each predictive col
        for col in cols_predictive:
            # compute slope
            data_slope = sdf.loc[df[col_date].between('2020-08-15', '2020-12-31', inclusive='both')]
            slope = get_slope_from_temporal_series(data_slope[col_date].dt.strftime('%y%j').astype(float),
                                                   data_slope[col])

            features.update({f'slope_end_year_{col}': slope})

            data_slope = sdf.loc[df[col_date].between('2020-01-1', '2020-3-3', inclusive='both')]
            slope = get_slope_from_temporal_series(data_slope[col_date].dt.strftime('%y%j').astype(float),
                                                   data_slope[col])

            features.update({f'slope_start_year_{col}': slope})

        all_features.append(features)
    df_features = pd.DataFrame(all_features)

    if for_train:
        # add plot type
        # generate a dict of plot_number and plot-type
        type_of_plots = pd.Series(df[plot_class].values, index=df[plot_nb]).to_dict()
        df_features['type'] = df_features[plot_nb].map(type_of_plots)
    return df_features
