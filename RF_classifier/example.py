import numpy as np
import pandas as pd

from RF_classifier.common import smooth_variables, normalise_cols
from RF_classifier.features import generate_features
from info import outputs


def get_example():
    # get stat and prepare features
    stats = pd.read_csv(outputs / 'stats/seg_rad50_sp20_test.csv', sep=',', parse_dates=['image_date_time_ksa'])
    stats = stats[stats['nbPixels'] >= 100]
    stats['VV_dB'] = 10 * np.log10(stats['VV_L'])
    stats['VH_dB'] = 10 * np.log10(stats['VH_L'])
    stats['VV_VH_dB'] = stats['VV_dB'] - stats['VH_dB']
    stats = stats.drop(['VV_L', 'VH_L'], axis='columns')
    stats = stats.sort_values(['label', 'image_date_time_ksa'])

    # generate more raw variables
    df = smooth_variables(stats, ['label', 'inc_class', 'year'], ['VV_dB', 'VH_dB', 'VV_VH_dB'], 2)

    # get features for all segments
    features = generate_features(df, 'label', 'ref_hand',
                                 ['VV_dB', 'VV_dB_smooth', 'VH_dB', 'VH_dB_smooth', 'VV_VH_dB', 'VV_VH_dB_smooth'],
                                 'image_date_time_ksa', clos_cmp=['VV_dB_smooth', 'VH_dB_smooth'])

    var_names = features.columns.to_list()
    var_names.remove('label')
    var_names.remove('ref_class')

    features = normalise_cols(features, var_names)

    return features, df, var_names
