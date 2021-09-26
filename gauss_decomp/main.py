import pandas as pd

from gauss_decomp.modelling import gaussian_modelling_ndvi_series


def apply_gauss_decomp(df, col, col_date, plot_id, gauss_nb, sigma):
    """
    Args:
        df (DataFrame): dataframe contains ndvi data
        col (str): name of col contains the date
        col_date (str): name of col contains the date
        plot_id (str): name of col contains the plot id

    Returns:
        (DataFrame): dataframe contains the gauss dcmp

    """
    all_df = []
    for name, sdf in df.groupby(by=[plot_id]):
        # apply modelling for VV
        ndvi_dcmp = gaussian_modelling_ndvi_series(sdf, col_date, col, gauss_nb, sigma)
        all_df.append(ndvi_dcmp)
    all_df = pd.concat(all_df)
    return all_df
