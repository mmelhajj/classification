import matplotlib.pyplot as plt

from RF_classifier.example import get_features
from info import outputs
from viualization.common import plot_temporal_evolution

# get example
_, df, _ = get_features()

# get all combinations
df['combi'] = df['type_1st_h'] + '/' + df['type_2nd_h']

fig, axes = plt.subplots(ncols=4, nrows=10, figsize=(15, 15), sharex='all')

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
        plot_temporal_evolution(x=date, y=vv_smooth_db, ax=axes[id, 0], y_label='', text_font_size=10,
                                xylabel_font_size=15, marker=None, ls='-', ylim=[-19, -9], title=f'{name}(VV)')

        plot_temporal_evolution(x=date, y=vh_smooth_db, ax=axes[id, 1], y_label='', text_font_size=10,
                                xylabel_font_size=15, marker=None, ls='-', ylim=[-25, -10], title=f'{name}(VH)')

        plot_temporal_evolution(x=date, y=vv_vh_smooth_db, ax=axes[id, 2], y_label='', text_font_size=10,
                                xylabel_font_size=15, marker=None, ls='-', ylim=[5, 15], title=f'{name}(VV-VH)')

        plot_temporal_evolution(x=date, y=ndvi, ax=axes[id, 3], y_label='', text_font_size=10,
                                xylabel_font_size=15, marker=None, ls='-', ylim=[0, 1], title=f'{name}(NDVI)')

        # axes[id, 0].text(date.values[-1], vv_smooth_db.values[-1], plot_id)
plt.savefig(f"{outputs}/figures/graph.png", bbox_inches='tight', pad_inches=0.1)
plt.close()
