import numpy as np
import pandas as pd

from RF_classifier.common import smooth_variables, gauss_decomp, normalise_cols
from RF_classifier.features import generate_features
from info import outputs


# if __name__ == '__main__':
#     # get example
#     _, df_query, _ = get_example()
#     df_template = pd.read_csv(outputs / 'template.csv', sep=',')
#     df = dtw_compute(df_query, 'name', ['VV_dB', 'VH_dB'], df_template, 'ref_class')


def get_example():
    # get stat and prepare features
    stats = pd.read_csv(outputs / 'stats/hand_map.csv', sep=',', parse_dates=['image_date_time_ksa'])
    stats = stats.loc[stats['image_date_time_ksa'].between('2015-01-01','2015-12-31')]
    stats = stats[stats['type_1st_h'] != 'not classified']
    # stats = stats[stats['nbPixels'] >= 100]
    stats['VV_dB'] = 10 * np.log10(stats['VV_L'])
    stats['VH_dB'] = 10 * np.log10(stats['VH_L'])
    stats['VV_VH_dB'] = stats['VV_dB'] - stats['VH_dB']
    stats = stats.drop(['VV_L', 'VH_L'], axis='columns')
    stats = stats.sort_values(['name', 'image_date_time_ksa'])

    # generate more raw variables
    df = smooth_variables(stats, ['name', 'inc_class', 'year'], ['VV_dB', 'VH_dB', 'VV_VH_dB'], 2)

    # scale data between 0 and 1
    df = normalise_cols(df, ['VV_dB', 'VH_dB', 'VV_dB_smooth', 'VH_dB_smooth'], 'name')

    # decompose into gauss
    df = gauss_decomp(df, ['VV_dB_smooth','VH_dB_smooth'], 'name')

    # # get dtw
    # df_template = pd.read_csv(outputs / 'template.csv', sep=',')
    # dtw = dtw_compute(df, 'name', ['VV_dB_smooth', 'VH_dB_smooth'], df_template, 'ref_class')

    # get features for all segments
    features = generate_features(df, 'name', 'type_1st_h', ['VV_dB_smooth', 'VH_dB_smooth', 'VV_VH_dB_smooth','VV_dB_smooth_decomp'],
                                 'image_date_time_ksa', clos_cmp=['VV_dB_smooth', 'VH_dB_smooth'])

    # features = features.merge(dtw, left_on='label', right_on='label')

    # normalize features between 0 and 1
    var_names = features.columns.to_list()
    var_names.remove('label')
    var_names.remove('ref_class')

    return features, df, var_names
