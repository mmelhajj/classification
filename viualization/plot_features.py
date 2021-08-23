import joblib
import matplotlib.pyplot as plt
import pandas as pd

from RF_classifier.example import get_example
from info import outputs, classfier
from viualization.common import plot_hist

# get example
features, _, var_names = get_example()

# load the model to extract importance and sort var by importance
model = joblib.load(classfier)
importances = model.feature_importances_
df = pd.DataFrame({'var': var_names, 'importances': importances})
df = df.sort_values(['importances'], ascending=False)
var_order_imp = df['var'].to_list()

# drop col and set index
features = features.drop(['label'], axis='columns')
features = features.set_index(['ref_class'])

# plot by importance
fig, axes = plt.subplots(ncols=10, nrows=4, figsize=(18, 15))
ax = axes.flatten()
for i, cl in enumerate(var_order_imp):
    for name, sdf in features.groupby(by=[features.index]):
        plot_hist(sdf[cl], 10, cl.replace('_dB', ''), '', '', ax[i], color=None, label=name, ylim=[0, 50], font=10,
                  rotation=0)
ax[0].legend(bbox_to_anchor=(0, 1))
plt.savefig(f"{outputs}/figures/a_variables_distribution.png", bbox_inches='tight', pad_inches=0.2)
