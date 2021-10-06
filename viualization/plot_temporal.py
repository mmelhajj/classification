import datetime

import matplotlib.pyplot as plt

from RF_classifier.example import get_features
from info import outputs
from viualization.common import plot_temporal_evolution

# get example
_, df, _ = get_features()

# get all combinations
df['combi'] = df['type_1st_h'] + '/' + df['type_2nd_h']

text = [['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'],
        ['i', 'j', 'k', 'l', 'm', 'n', 'o', 'p'],
        ['q', 'r', 's', 't', 'u', 'x', 'y', 'z'],
        ['a1', 'b1', 'c1', 'd1', 'e1', 'f1', 'g1', 'h1']]

fig, axes = plt.subplots(ncols=8, nrows=4, figsize=(30, 15), sharex='all', sharey='row')
for id, (name, sdf) in enumerate(df.groupby(by=['combi'])):
    for plot_id, s_sdf in sdf.groupby(by=['name']):
        # convert to dB
        date = s_sdf['image_date_time_ksa']
        vv_smooth_db = s_sdf['VV_dB_smooth']
        vh_smooth_db = s_sdf['VH_dB_smooth']
        vv_vh_smooth_db = s_sdf['VV_VH_dB_smooth']
        ndvi = s_sdf['ndvi_smooth']
        # plot = sdf['ref_hand'].unique()[0]

        # plot for VV
        plot_temporal_evolution(x=date, y=vv_smooth_db, ax=axes[0, id], y_label='VV (dB)', text_font_size=20,
                                xylabel_font_size=15, marker=None, ls='-', ylim=[-19, -8], title=f'{name}')

        plot_temporal_evolution(x=date, y=vh_smooth_db, ax=axes[1, id], y_label='VH(dB)', text_font_size=20,
                                xylabel_font_size=15, marker=None, ls='-', ylim=[-26, -10], title=f'{name}')

        plot_temporal_evolution(x=date, y=vv_vh_smooth_db, ax=axes[2, id], y_label='VV-VH (dB)', text_font_size=20,
                                xylabel_font_size=15, marker=None, ls='-', ylim=[5, 15], title=f'{name}')

        plot_temporal_evolution(x=date, y=ndvi, ax=axes[3, id], y_label='NDVI', text_font_size=20,
                                xylabel_font_size=15, marker=None, ls='-', ylim=[0, 1], title=f'{name}')

        axes[0, id].text(datetime.datetime(2014, 11, 10), -10, text[0][id], fontsize=30)
        axes[1, id].text(datetime.datetime(2014, 11, 10), -12, text[1][id], fontsize=30)
        axes[2, id].text(datetime.datetime(2014, 11, 10), 13, text[2][id], fontsize=30)
        axes[3, id].text(datetime.datetime(2014, 11, 10), 0.8, text[3][id], fontsize=30)

        # axes[id, 0].text(date.values[-1], vv_smooth_db.values[-1], plot_id)
plt.savefig(f"{outputs}/figures/graph.png", bbox_inches='tight', pad_inches=0.1)
plt.close()
