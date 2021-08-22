import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from RF_classifier.common import generate_variables
from RF_classifier.features import generate_features
from info import outputs

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

# define inputs and output and split
df_x = features.drop(['label', 'ref_class'], axis='columns')
df_y = features['ref_class']
X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.5, random_state=0)

# fit RF regressor classifier: https://towardsdatascience.com/classification-with-random-forests-in-python-29b8381680ed
model = RandomForestClassifier(n_estimators=100, random_state=24)
model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test)

# get score
fscore = f1_score(y_test, y_pred, average='macro')
print(f'Fscore : {fscore}*100')

# get and plot importance
importance = model.feature_importances_
importance = pd.DataFrame(importance, columns=['importance'], index=X_train.columns)
importance = importance.sort_values(['importance'])

fig_, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10), sharex='all')
ax.barh(importance.index, importance['importance'])

# # fmt axes
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=10, rotation=30)
ax.set_xlabel('Performance', fontsize=20)

# save figure
plt.savefig(f"{outputs}/importance.png", dpi=300, bbox_inches='tight', pad_inches=0.1)

plt.show()
