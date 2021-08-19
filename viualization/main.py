import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import dates
from RF_classifier.common import gaussian_smoothing
from RF_classifier.common import get_data_from_shape, set_training
from info import outputs
from viualization.common import plot_temporal_evolution

shape = outputs / 'seg_rad50_sp20.shp'
df_label, _ = get_data_from_shape(shape, ['label', 'type'], save_csv=True, path_csv=outputs,
                                  name_csv='training_data.csv')
df_label = df_label.dropna()

training = set_training(outputs / 'stats/seg_rad50_sp20.csv', df_label, 'label', 'label')
training['image_date'] = pd.to_datetime(training['image_date'])
training = training.sort_values(['label', 'image_date'])

df = []
for (name, inc, year), sdf in training.groupby(by=['label', 'inc_class', 'year']):
    # apply smoothing only
    s_sdf = gaussian_smoothing(sdf, 'VV_L', 2)
    s_sdf = gaussian_smoothing(s_sdf, 'VH_L', 2)
    df.append(s_sdf)
df = pd.concat(df)

for (name, inc, year), sdf in df.groupby(by=['label', 'inc_class', 'year']):
    fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(15, 15), sharex='all')
    axes = axes.flatten()
    # convert to dB
    date = sdf['image_date']
    vv_db = 10 * np.log10(sdf['VV_L'])
    vh_db = 10 * np.log10(sdf['VH_L'])
    vv_smooth_db = 10 * np.log10(sdf['VV_L_smooth'])
    vh_smooth_db = 10 * np.log10(sdf['VH_L_smooth'])
    plot = sdf['type'].unique()[0]

    # plot for VV
    plot_temporal_evolution(x=date, y=vv_db, ax=axes[0], y_label='VV(dB)', text_font_size=30, xylabel_font_size=30,
                            ylim=[-30, -5], marker='o', ls='-')

    plot_temporal_evolution(x=date, y=vv_smooth_db, ax=axes[0], y_label='VV(dB)', text_font_size=30,
                            xylabel_font_size=30, ylim=[-30, -5], marker='d')

    plot_temporal_evolution(x=date, y=vh_db, ax=axes[1], text_font_size=30, y_label='VH(dB)', xylabel_font_size=30,
                            ylim=[-30, -10], marker='o', ls='-')

    plot_temporal_evolution(x=date, y=vh_smooth_db, ax=axes[1], text_font_size=30, y_label='VH(dB)',
                            xylabel_font_size=30, ylim=[-30, -10], marker='d')

    # axes[1].xaxis.set_major_locator(dates.MonthLocator(interval=1))
    # axes[1].xaxis.set_major_formatter(dates.DateFormatter('%Y-%m'))
    # set title
    fig.suptitle(f'Plot : {plot}', fontsize=40)

    # save figure
    fig_name = inc + "_" + plot + "_" + name.astype(str)
    plt.savefig(f"{outputs}/figures/{fig_name}", bbox_inches='tight', pad_inches=0.1)

    plt.close()
