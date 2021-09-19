import matplotlib.pyplot as plt
from RF_classifier.common import normalise_cols
from RF_classifier.example import get_example
from info import outputs
from viualization.common import plot_temporal_evolution

# get example
_, df, _ = get_example()

fig, axes = plt.subplots(ncols=2, nrows=4, figsize=(15, 15), sharex='all', sharey='all')
axes = axes.flatten()

for id, (name, sdf) in enumerate(df.groupby(by=['type_1st_h'])):
    for _, s_sdf in sdf.groupby(by=['name']):
        # convert to dB
        date = s_sdf['image_date_time_ksa']
        vv_db = s_sdf['VH_dB']
        vv_smooth_db = s_sdf['VH_dB_smooth']
        # plot = sdf['ref_hand'].unique()[0]

        # plot for VV
        plot_temporal_evolution(x=date, y=vv_smooth_db, ax=axes[id], y_label='VV(dB)', text_font_size=30,
                                xylabel_font_size=30,
                                marker='o', ls='-', ylim=None)

    axes[id].set_title(name)

    # plot_temporal_evolution(x=date, y=vh_db, ax=axes[1], text_font_size=30, y_label='VH(dB)', xylabel_font_size=30,
    #                         ylim=None, marker='o', ls='-')

    # axes[1].xaxis.set_major_locator(dates.MonthLocator(interval=1))
    # axes[1].xaxis.set_major_formatter(dates.DateFormatter('%Y-%m'))
    # set title
    # fig.suptitle(f'Plot : {plot}', fontsize=40)

    # save figure
    # fig_name = inc + "_" + plot + "_" + name.astype(str)
plt.savefig(f"{outputs}/figures/graph.png", bbox_inches='tight', pad_inches=0.1)
plt.show()
plt.close()
