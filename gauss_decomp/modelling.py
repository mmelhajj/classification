"""filter LAI verde with Gaussian smooth filter and removes aberrant LAI verde points
"""

from scipy.optimize import curve_fit

from gauss_decomp.gaussian_equations import gaussian_smoothing_method, one_gaussian, two_gaussian


def gaussian_modelling_ndvi_series(df, date_column, col_data_to_filter, maximum_gauss_nb, sigma):
    # # give name to col with modelled data
    # output_col_name = col_data_to_filter + f'_modelled_{maximum_gauss_nb}_gauss'

    # reset index
    df = df.reset_index()

    # create a cols with DOY from first LAI verde date
    df['days'] = (df[date_column] - df[date_column].min()).astype('timedelta64[D]')

    # smooth SAR scattering
    smooth, peak_pos = gaussian_smoothing_method(df, col_data_to_filter, maximum_gauss_nb, sigma)

    # affine _peak pos
    peaks_dates = df.iloc[peak_pos][date_column].diff().dropna().dt.days.values[0]
    if peaks_dates < 30:
        peak_pos = peak_pos[0]

    # save the smooth profile
    df[f'gauss_smooth_{col_data_to_filter}'] = smooth

    # Model using only one gaussian
    popt = None
    if len(peak_pos) == 1:
        popt, pcov = curve_fit(one_gaussian, df['days'], df[col_data_to_filter],
                               p0=[smooth[peak_pos[0]], df['days'].iloc[peak_pos].values[0], 90])
        # # Apply modelling
        # df[output_col_name] = one_gaussian(df['days'], *popt)

    if len(peak_pos) == 2:
        popt, pcov = curve_fit(two_gaussian, df['days'], df[col_data_to_filter],
                               p0=[smooth[peak_pos[0]], df['days'].iloc[peak_pos].values[0], 30,
                                   smooth[peak_pos[1]], df['days'].iloc[peak_pos].values[1], 30])
        # # Apply modelling
        # df[output_col_name] = two_gaussian(df['days'], *popt)

    elif len(peak_pos) > 2:
        raise ValueError("the desired nb of gaussian should be lower than 3")

    return popt.reshape((-1, 3))
