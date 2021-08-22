import matplotlib.pyplot as plt
import pandas as pd

from RF_classifier.common import generate_variables
from RF_classifier.features import generate_features
from info import outputs
from viualization.common import plot_hist

# get stat and prepare features
stats = pd.read_csv(outputs / 'stats/seg_rad50_sp20_test.csv', sep=',', parse_dates=['image_date_time_ksa'])
stats['VV_VH_L'] = stats['VV_L'] / stats['VH_L']
stats = stats.sort_values(['label', 'image_date_time_ksa'])

# generate more raw variables
df = generate_variables(stats, ['label', 'inc_class', 'year'])

# get features for all segments
features = generate_features(df, 'label', 'ref_hand',
                             ['VV_L', 'VV_L_smooth', 'VH_L', 'VH_L_smooth', 'VV_VH_L', 'VV_VH_L_smooth'],
                             'image_date_time_ksa')

# drop col and set index
features = features.drop(['label'], axis='columns')
features = features.set_index(['ref_class'])

fig, axes = plt.subplots(ncols=6, nrows=5, figsize=(15, 15))
ax = axes.flatten()
for i, cl in enumerate(features.columns):
    for name, sdf in features.groupby(by=[features.index]):
        plot_hist(sdf[cl], 10, '', cl, '', ax[i], color=None, label=name, ylim=[0, 50], font=5, rotation=0)
ax[-1].legend()
plt.savefig(f"{outputs}/figures/a_variables_distribution.png", bbox_inches='tight', pad_inches=0.1)
