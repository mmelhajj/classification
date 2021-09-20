import numpy as np
import pandas as pd

from RF_classifier.common import smooth_variables, normalise_cols
from info import outputs


def get_dtw_example():
    # get stat and prepare features
    stats = pd.read_csv(outputs / 'stats/hand_map.csv', sep=',', parse_dates=['image_date_time_ksa'])
    stats = stats[stats['type_1st_h'] != 'not classified']
    # stats = stats[stats['nbPixels'] >= 100]
    stats['VV_dB'] = 10 * np.log10(stats['VV_L'])
    stats['VH_dB'] = 10 * np.log10(stats['VH_L'])
    stats['VV_VH_dB'] = stats['VV_dB'] - stats['VH_dB']
    stats = stats.drop(['VV_L', 'VH_L'], axis='columns')
    stats = stats.sort_values(['name', 'image_date_time_ksa'])

    # generate more raw variables
    df = smooth_variables(stats, ['name', 'inc_class', 'year'], ['VV_dB', 'VH_dB', 'VV_VH_dB'], 2)

    # # scale data between 0 and 1
    # df = normalise_cols(df, ['VV_dB', 'VH_dB', 'VV_dB_smooth', 'VH_dB_smooth', 'VV_VH_dB_smooth'], 'name')

    # get dtw
    df_template = pd.read_csv(outputs / 'template.csv', sep=',')

    return df, df_template
