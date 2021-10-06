"""extracts template for each crop and save in df"""

import matplotlib.pyplot as plt
import pandas as pd

from RF_classifier.example import prepare_data
from info import outputs
from viualization.common import plot_temporal_evolution

# get example
df = prepare_data()

df['combi'] = df['type_1st_h'] + "_" + df['type_2nd_h']

# # define date to extract the template
# cycle_start_end = {'wheat': {'start': datetime(2014, 1, 1), 'end': datetime(2015, 12, 31)},
#                    'alfalfa': {'start': datetime(2014, 1, 1), 'end': datetime(2015, 12, 31)},
#                    'barely': {'start': datetime(2014, 1, 1), 'end': datetime(2015, 12, 31)},
#                    'fruit': {'start': datetime(2014, 1, 1), 'end': datetime(2015, 12, 31)},
#                    'olive': {'start': datetime(2014, 1, 1), 'end': datetime(2015, 12, 31)},
#                    'onion': {'start': datetime(2014, 1, 1), 'end': datetime(2015, 12, 31)},
#                    'potato': {'start': datetime(2014, 1, 1), 'end': datetime(2015, 12, 31)},
#                    'corn': {'start': datetime(2014, 1, 1), 'end': datetime(2015, 12, 31)}}

# average the profile for each crop type separately
all_df = []
for name, sdf in df.groupby(by=['combi']):
    sub_sdf = sdf[
        ['projectedInc', 'image_date_time_ksa', 'VV_dB', 'VH_dB', 'VV_VH_dB', 'VV_dB_smooth', 'VH_dB_smooth',
         'VV_VH_dB_smooth', 'ndvi_smooth']].groupby(by=['image_date_time_ksa']).mean()
    sub_sdf['ref_class'] = name
    all_df.append(sub_sdf)

df = pd.concat(all_df)

# save in df
with open(outputs / 'template.csv', 'w') as temp:
    df.to_csv(temp, index=True, line_terminator='\n')

# plot to verify
if __name__ == '__main__':
    n_rows = len(df['ref_class'].unique())
    fig, axes = plt.subplots(ncols=4, nrows=n_rows, figsize=(15, 15), sharex='all')
    for id, (name, sdf) in enumerate(df.groupby(by=['ref_class'])):
        print(name, id)
        # set X ad Ys
        date = sdf.index
        vv_smooth_db = sdf['VV_dB_smooth']

        vh_smooth_db = sdf['VH_dB_smooth']

        vv_vh_smooth_db = sdf['VV_VH_dB_smooth']

        ndvi_smooth = sdf['ndvi_smooth']

        # plot for VV
        plot_temporal_evolution(x=date, y=vv_smooth_db, ax=axes[id, 0], y_label='', text_font_size=30,
                                xylabel_font_size=30, marker='o', ls='-', ylim=None, label='VV_sm')

        # # plot for VH
        plot_temporal_evolution(x=date, y=vh_smooth_db, ax=axes[id, 1], y_label='', text_font_size=30,
                                xylabel_font_size=30, marker='o', ls='-', ylim=None, label='VH_sm')
        # # plot for VV-VH
        plot_temporal_evolution(x=date, y=vv_vh_smooth_db, ax=axes[id, 2], y_label='', text_font_size=30,
                                xylabel_font_size=30, marker='o', ls='-', ylim=None, label='VV-VH_sm')
        # # plot for ndvi
        plot_temporal_evolution(x=date, y=ndvi_smooth, ax=axes[id, 3], y_label='', text_font_size=30,
                                xylabel_font_size=30, marker='o', ls='-', ylim=None, label='ndvi_smooth')

        for x in range(4):
            axes[id, x].set_title(name)

    plt.show()
