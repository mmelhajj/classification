import geopandas as gpd
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from RF_classifier.common import generate_variables
from RF_classifier.common import get_data_from_shape, set_training
from RF_classifier.features import generate_features
from info import outputs

# get stat and prepare features
stats = pd.read_csv(outputs / 'stats/seg_rad50_sp20.csv', sep=',', parse_dates=['image_date_time_ksa'])
stats['VV_VH_L'] = stats['VV_L'] - stats['VH_L']

# generate more raw variables
df = generate_variables(stats, ['label', 'inc_class', 'year'])
# get features for all segments
X_est = generate_features(df, 'label', 'type',
                          ['VV_L', 'VV_L_smooth', 'VH_L', 'VH_L_smooth', 'VV_VH_L', 'VV_VH_L_smooth'],
                          'image_date_time_ksa', for_train=False)

# get features for training segments
# open the segmentation file, it should contain class name for some segments
shape = outputs / 'seg_rad50_sp20.shp'
df_label, _ = get_data_from_shape(shape, ['label', 'type'], save_csv=True, path_csv=outputs,
                                  name_csv='training_data.csv')
df_label = df_label.dropna()

# isolate df rows with knowen class
training = set_training(df, df_label, 'label', 'label')
training = training.sort_values(['label', 'image_date_time_ksa'])

# get features for training segments
X_train, Y_target = generate_features(training, 'label', 'type',
                                      ['VV_L', 'VV_L_smooth', 'VH_L', 'VH_L_smooth', 'VV_VH_L', 'VV_VH_L_smooth'],
                                      'image_date_time_ksa')

# random forest classifier
# https://towardsdatascience.com/classification-with-random-forests-in-python-29b8381680ed
model = RandomForestClassifier(n_estimators=100, random_state=24)

# fit the model
model.fit(X_train, Y_target)

# pred
X_est['pred'] = model.predict(X_est)

# map, read the segmentation shape file and add pred, and save as shape
gdf = gpd.read_file(shape)
gdf = gdf.merge(X_est, left_on='label', right_on='label', how='inner')
gdf.to_file(driver='ESRI Shapefile', filename=outputs / "result.shp")

# generate confusion matrix
df = gdf[['ref_hand', 'pred']]
df = df.dropna()
conf_matrix = confusion_matrix(df['ref_hand'], df['pred'], labels=df['ref_hand'].unique())
fscore = f1_score(df['ref_hand'], df['pred'], average='macro')

print('fscore:', fscore)
