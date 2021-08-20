import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from RF_classifier.common import generate_variables
from RF_classifier.common import get_data_from_shape, set_training
from info import outputs
from viualization.common import plot_temporal_evolution

# get stat and prepare features
stats = pd.read_csv(outputs / 'stats/seg_rad50_sp20.csv', sep=',', parse_dates=['image_date_time_ksa'])

# generate more raw variables
df = generate_variables(stats, ['label', 'inc_class', 'year'])

# get features for training segments
# open the segmentation file, it should contain class name for some segments
shape = outputs / 'seg_rad50_sp20.shp'
df_label, _ = get_data_from_shape(shape, ['label', 'type'], save_csv=True, path_csv=outputs,
                                  name_csv='training_data.csv')
df_label = df_label.dropna()

# isolate df rows with knowen class
training = set_training(df, df_label, 'label', 'label')
training = training.sort_values(['label', 'image_date_time_ksa'])

for (name, inc, year), sdf in training.groupby(by=['label', 'inc_class', 'year']):
    fig, axes = plt.subplots(ncols=1, nrows=3, figsize=(15, 15), sharex='all')
    axes = axes.flatten()
    # convert to dB
    date = sdf['image_date_time_ksa']
    vv_db = 10 * np.log10(sdf['VV_L'])
    vh_db = 10 * np.log10(sdf['VH_L'])
    vv_smooth_db = 10 * np.log10(sdf['VV_L_smooth'])
    vh_smooth_db = 10 * np.log10(sdf['VH_L_smooth'])
    plot = sdf['type'].unique()[0]

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
