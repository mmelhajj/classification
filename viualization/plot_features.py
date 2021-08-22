import matplotlib.pyplot as plt

from RF_classifier.example import get_example
from info import outputs
from viualization.common import plot_hist

# get example
features, _ = get_example()

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
