"""extracts template for each crop and save in df"""
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd

from RF_classifier.example import get_example
from info import outputs
from viualization.common import plot_temporal_evolution

# get example
_, df, _ = get_example()

# define date to extract the template
cycle_start_end = {'wheat': {'start': datetime(2014, 1, 1), 'end': datetime(2015, 12, 31)},
                   'alfaalfa': {'start': datetime(2014, 1, 1), 'end': datetime(2015, 12, 31)},
                   'barely': {'start': datetime(2014, 1, 1), 'end': datetime(2015, 12, 31)},
                   'fruit': {'start': datetime(2014, 1, 1), 'end': datetime(2015, 12, 31)},
                   'olive': {'start': datetime(2014, 1, 1), 'end': datetime(2015, 12, 31)},
                   'onion': {'start': datetime(2014, 1, 1), 'end': datetime(2015, 12, 31)},
                   'potato': {'start': datetime(2014, 1, 1), 'end': datetime(2015, 12, 31)},
                   'sudanese corn': {'start': datetime(2014, 1, 1), 'end': datetime(2015, 12, 31)}}

# average the profile for each crop type separately
all_df = []
for name, sdf in df.groupby(by=['type_1st_h']):
    sub_sdf = sdf.loc[sdf['image_date_time_ksa'].between(cycle_start_end[name]['start'], cycle_start_end[name]['end'])]
    sub_sdf = sub_sdf[
        ['projectedInc', 'image_date_time_ksa', 'VV_dB', 'VH_dB', 'VV_VH_dB', 'VV_dB_smooth', 'VH_dB_smooth',
         'VV_VH_dB_smooth']].groupby(by=['image_date_time_ksa']).mean()
    sub_sdf['ref_class'] = name
    all_df.append(sub_sdf)

df = pd.concat(all_df)

# save in df
with open(outputs / 'template.csv', 'w') as temp:
    df.to_csv(temp, index=True, line_terminator='\n')

# plot to verify
if __name__ == '__main__':
    fig, axes = plt.subplots(ncols=2, nrows=4, figsize=(15, 15), sharex='all', sharey='all')
    axes = axes.flatten()
    for id, (name, sdf) in enumerate(df.groupby(by=['ref_class'])):
        date = sdf.index
        vv_db = sdf['VV_dB']
        vv_smooth_db = sdf['VV_dB_smooth']

        # plot for VV
        plot_temporal_evolution(x=date, y=vv_smooth_db, ax=axes[id], y_label='VV(dB)', text_font_size=30,
                                xylabel_font_size=30, marker='o', ls='-', ylim=None)
        axes[id].set_title(name)
    plt.show()
