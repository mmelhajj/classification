import numpy as np
import pandas as pd
from info import outputs
from RF_classifier.common import smooth_variables, normalise_cols
from RF_classifier.example import prepare_data


def get_dtw_example():
    stats = prepare_data()

    # generate more raw variables
    df = smooth_variables(stats, ['name', 'inc_class', 'year'], ['VV_dB', 'VH_dB', 'VV_VH_dB'], 2)

    # # scale data between 0 and 1
    # df = normalise_cols(df, ['VV_dB', 'VH_dB', 'VV_dB_smooth', 'VH_dB_smooth', 'VV_VH_dB_smooth'], 'name')

    # get dtw
    df_template = pd.read_csv(outputs / 'template.csv', sep=',')

    return df, df_template
