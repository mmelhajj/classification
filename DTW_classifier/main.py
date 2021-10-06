from dtw_classifier.common import dtw_compute
from dtw_classifier.get_example import get_dtw_example

df, df_template = get_dtw_example()
dtw = dtw_compute(df, 'name', ['VV_dB_smooth'], df_template, 'ref_class')
