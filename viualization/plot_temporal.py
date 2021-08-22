import matplotlib.pyplot as plt

from RF_classifier.example import get_example
from info import outputs
from viualization.common import plot_temporal_evolution

# get example
_, df = get_example()

for (name, inc, year), sdf in df.groupby(by=['label', 'inc_class', 'year']):
    fig, axes = plt.subplots(ncols=1, nrows=3, figsize=(15, 15), sharex='all')
    axes = axes.flatten()
    # convert to dB
    date = sdf['image_date_time_ksa']
    vv_db = sdf['VV_dB']
    vh_db = sdf['VH_dB']
    vv_smooth_db = sdf['VV_dB_smooth']
    vh_smooth_db = sdf['VH_dB_smooth']
    plot = sdf['ref_hand'].unique()[0]

    # plot for VV
    plot_temporal_evolution(x=date, y=vv_db, ax=axes[0], y_label='VV(dB)', text_font_size=30, xylabel_font_size=30,
                            ylim=[-25, -5], marker='o', ls='-')

    plot_temporal_evolution(x=date, y=vv_smooth_db, ax=axes[0], y_label='VV(dB)', text_font_size=30,
                            xylabel_font_size=30, ylim=[-25, -5], marker='d')

    plot_temporal_evolution(x=date, y=vh_db, ax=axes[1], text_font_size=30, y_label='VH(dB)', xylabel_font_size=30,
                            ylim=[-30, -10], marker='o', ls='-')

    plot_temporal_evolution(x=date, y=vh_smooth_db, ax=axes[1], text_font_size=30, y_label='VH(dB)',
                            xylabel_font_size=30, ylim=[-30, -10], marker='d')

    plot_temporal_evolution(x=date, y=vv_smooth_db - vh_smooth_db, ax=axes[2], y_label='VV/VH(dB)', text_font_size=30,
                            xylabel_font_size=30, ylim=[0, 15], marker='d')
    # axes[1].xaxis.set_major_locator(dates.MonthLocator(interval=1))
    # axes[1].xaxis.set_major_formatter(dates.DateFormatter('%Y-%m'))
    # set title
    fig.suptitle(f'Plot : {plot}', fontsize=40)

    # save figure
    fig_name = inc + "_" + plot + "_" + name.astype(str)
    plt.savefig(f"{outputs}/figures/{fig_name}", bbox_inches='tight', pad_inches=0.1)

    plt.close()
