import matplotlib.pyplot as plt

from RF_classifier.example import get_example
from info import outputs
from viualization.common import plot_temporal_evolution

# get example
_, df, _ = get_example()

fig, axes = plt.subplots(ncols=3, nrows=6, figsize=(15, 15), sharex='all', sharey='all')
# axes = axes.flatten()

for id, (name, sdf) in enumerate(df.groupby(by=['type_1st_h'])):
    for plot_id, s_sdf in sdf.groupby(by=['name']):
        # convert to dB
        date = s_sdf['image_date_time_ksa']
        vv_smooth_db = s_sdf['VV_dB_smooth']
        vh_smooth_db = s_sdf['VH_dB_smooth']
        vv_vh_smooth_db = s_sdf['VV_VH_dB_smooth']
        # plot = sdf['ref_hand'].unique()[0]

        # plot for VV
        plot_temporal_evolution(x=date, y=vv_smooth_db, ax=axes[id, 0], y_label='', text_font_size=10,
                                xylabel_font_size=30, marker=None, ls='-', ylim=None, title=name)

        plot_temporal_evolution(x=date, y=vh_smooth_db, ax=axes[id, 1], y_label='', text_font_size=10,
                                xylabel_font_size=30, marker=None, ls='-', ylim=None, title=name)

        plot_temporal_evolution(x=date, y=vv_vh_smooth_db, ax=axes[id, 2], y_label='', text_font_size=10,
                                xylabel_font_size=30, marker=None, ls='-', ylim=None, title=name)
        axes[id, 0].text(date.values[-1], vv_smooth_db.values[-1], plot_id)
plt.savefig(f"{outputs}/figures/graph.png", bbox_inches='tight', pad_inches=0.1)
plt.show()
plt.close()
