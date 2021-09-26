import numpy as np
import pandas as pd

from RF_classifier.common import smooth_variables, normalise_cols
from RF_classifier.features import generate_features
from info import outputs
from gauss_decomp.modelling import gaussian_modelling_ndvi_series


def get_example():
    # Prepare for features
    # read SAR data
    stats = pd.read_csv(outputs / 'stats/hand_map.csv', sep=',', parse_dates=['image_date_time_ksa'])
    stats = stats.loc[stats['image_date_time_ksa'].between('2014-11-01', '2015-12-01')]

    # add the ndvi to SAR data date by interpolation
    ndvi = pd.read_csv(outputs / 'stats/hand_map_ndvi.csv', sep=',', parse_dates=['image_date_time_ksa'])
    ndvi = ndvi[['image_date_time_ksa', 'name', 'ndvi']]
    # to create nan when the NDVO do not match SAR time
    ndvi = ndvi.merge(stats[['image_date_time_ksa', 'name']], left_on=['image_date_time_ksa', 'name'],
                      right_on=['image_date_time_ksa', 'name'], how='outer')
    # sort value and add inted to interpolate
    ndvi = ndvi.sort_values(['name', 'image_date_time_ksa'])
    ndvi = ndvi.set_index('image_date_time_ksa')

    all_df = []
    for name, sdf in ndvi.groupby(by=['name']):
        sdf = sdf.interpolate(method='linear')
        all_df.append(sdf)
    all_df = pd.concat(all_df)
    all_df = all_df.reset_index()

    # join interpolated ndvi and SAR
    prev_nb = len(stats)
    stats = all_df.merge(stats, left_on=['name', 'image_date_time_ksa'], right_on=['name', 'image_date_time_ksa'],
                         how='inner')

    assert len(stats) == prev_nb

    # delete non cultivated plots
    all_df = []
    for name, sdf in stats.groupby(by=['name']):
        max_ndvi = sdf['ndvi'].max()
        if max_ndvi > 0.3:
            all_df.append(sdf)
    all_df = pd.concat(all_df)
    stats = all_df.reset_index()

    stats = stats[stats['type_1st_h'] != 'not classified']
    stats = stats[stats['type_1st_h'] != 'fruit']
    stats = stats[stats['type_1st_h'] != 'olive']

    # stats = stats[stats['nbPixels'] >= 100]
    stats['VV_dB'] = 10 * np.log10(stats['VV_L'])
    stats['VH_dB'] = 10 * np.log10(stats['VH_L'])
    stats['VV_VH_dB'] = stats['VV_dB'] - stats['VH_dB']
    stats = stats.drop(['VV_L', 'VH_L'], axis='columns')
    stats = stats.sort_values(['name', 'image_date_time_ksa'])

    # get season peak
    df_peaks = []
    peaks = {}
    for name, sdf in stats.groupby(by=['name']):
        try:
            sdf = sdf.loc[sdf['image_date_time_ksa'].between('2015-1-01', '2015-12-31')]
            peaks_pos = gaussian_modelling_ndvi_series(sdf, 'image_date_time_ksa', 'ndvi', 2, 0.001)
            peaks_pos = peaks_pos[:, 1]
            if len(peaks_pos) == 2:
                peaks.update({'name': name, 'peak1': peaks_pos[0], 'peak2': peaks_pos[1]})
            else:
                peaks.update({'name': name, 'peak1': peaks_pos[0]})
        except:
            print(name)


    # generate more raw variables
    df = smooth_variables(stats, ['name', 'inc_class', 'year'], ['VV_dB', 'VH_dB', 'VV_VH_dB', 'ndvi'], 2)

    # # scale data between 0 and 1
    # df = normalise_cols(df, ['VV_dB', 'VH_dB', 'VV_dB_smooth', 'VH_dB_smooth', 'VV_VH_dB_smooth', 'ndvi_smooth'],
    #                     'name')

    df = df[
        ['name', 'VV_dB_smooth', 'VH_dB_smooth', 'VV_VH_dB_smooth', 'type_1st_h', 'image_date_time_ksa', 'ndvi_smooth']]

    # # decompose into gauss
    # df = gauss_decomp(df, ['VV_dB_smooth', 'VH_dB_smooth', 'VV_VH_dB_smooth'], 'name')

    # # get dtw
    # df_template = pd.read_csv(outputs / 'template.csv', sep=',')
    # dtw = dtw_compute(df, 'name', ['VV_dB_smooth', 'VH_dB_smooth'], df_template, 'ref_class')

    # get features for all segments
    features = generate_features(df, 'name', 'type_1st_h',
                                 ['VV_dB_smooth', 'VH_dB_smooth', 'VV_VH_dB_smooth', 'ndvi_smooth'],
                                 'image_date_time_ksa', clos_cmp=['VV_dB_smooth', 'VH_dB_smooth'])

    # features = features.merge(dtw, left_on='label', right_on='label')

    # normalize features between 0 and 1
    var_names = features.columns.to_list()
    var_names.remove('label')
    var_names.remove('ref_class')

    return features, df, var_names
