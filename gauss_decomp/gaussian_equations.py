import numpy as np
from scipy.ndimage import gaussian_filter


def one_gaussian(x, amp1, pos1, sigma1):
    """
    Args:
        x (float): array element
        amp1 (float): amplitude of the gaussian
        pos1 (float): position of the peak
        sigma1 (float): sigma of the gaussian

    Returns:

    """
    res = amp1 * np.exp(-(x - pos1) ** 2 / (2 * sigma1 ** 2))
    return res


def two_gaussian(x, amp1, pos1, sigma1, amp2, pos2, sigma2):
    """
    Args:
        x (float): array element
        amp1 (float): amplitude of the first gaussian
        pos1 (float): position of the first peak
        sigma1 (float): sigma of the first gaussian
        amp2 (float): amplitude of the second gaussian
        pos2 (float): position of the second peak
        sigma2 (float): sigma of the second gaussian

    Returns:

    """
    res = amp1 * np.exp(-(x - pos1) ** 2 / (2 * sigma1 ** 2)) + amp2 * np.exp(-(x - pos2) ** 2 / (2 * sigma2 ** 2))
    return res


def gaussian_smoothing_method(df, col_data_to_smooth, maximum_gauss_nb=None, sigma=None):
    """
    Args:
        df (DataFrame): dataframe of data
        col_data_to_smooth (str): name of the col to smooth
        maximum_gauss_nb (int): maximum number of Gaussian needed, it is equal to the peak position (see cond),
        the smoothing will continue until the desired number is reached
        sigma (float): sigma of the Gaussian template that we use to smooth

    Returns:
        (array, array) smoothed values
    """
    # initiate the peak_nb
    nb_peak = maximum_gauss_nb + 1

    # initiate outputs
    smooth = []
    peak_pos = []

    # smooth until get the desired peak nb
    while nb_peak != maximum_gauss_nb:
        sigma += 0.05
        smooth = gaussian_filter(df[col_data_to_smooth], sigma)

        nb_peak = 0
        peak_pos = []
        for index, (item1, item2, item3) in enumerate(zip(smooth, smooth[1:], smooth[2:])):
            if item1 < item2 > item3:
                nb_peak += 1
                peak_pos.append(index)

    if len(smooth) == 0 or len(peak_pos) == 0:
        raise ValueError("smooth operation failed")

    return smooth, peak_pos
