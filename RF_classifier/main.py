import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import geopandas as gpd
from RF_classifier.common import gaussian_smoothing
from RF_classifier.common import get_data_from_shape, set_training
from RF_classifier.features import generate_features
from info import outputs

# get stat and prepare features
stats = pd.read_csv(outputs / 'stats/seg_rad50_sp20.csv', sep=',', parse_dates=['image_date_time_ksa'])
# add more variables
df = []
for (name, inc, year), sdf in stats.groupby(by=['label', 'inc_class', 'year']):
    # apply smoothing only
    s_sdf = gaussian_smoothing(sdf, 'VV_L', 2)
    s_sdf = gaussian_smoothing(s_sdf, 'VH_L', 2)
    df.append(s_sdf)
df = pd.concat(df)

features_est = generate_features(df, 'label', 'type', ['VV_L', 'VV_L_smooth', 'VH_L', 'VH_L_smooth'],
                                 'image_date_time_ksa', for_train=False)

# set training data, data from previous df with knwo class
# open the shape that contain class name
shape = outputs / 'seg_rad50_sp20.shp'
df_label, _ = get_data_from_shape(shape, ['label', 'type'], save_csv=True, path_csv=outputs,
                                  name_csv='training_data.csv')
df_label = df_label.dropna()

training = set_training(df, df_label, 'label', 'label')
training = training.sort_values(['label', 'image_date'])

features_train = generate_features(training, 'label', 'type', ['VV_L', 'VV_L_smooth', 'VH_L', 'VH_L_smooth'],
                                   'image_date_time_ksa')

x = 1

# random forest classifier

# model = RandomForestClassifier(n_estimators=100, random_state=24)
# https://towardsdatascience.com/classification-with-random-forests-in-python-29b8381680ed
model = RandomForestClassifier(n_estimators=100, random_state=24)

X_train = features_train[['slope_end_year_VV_L', 'slope_start_year_VV_L',
                          'slope_end_year_VV_L_smooth', 'slope_start_year_VV_L_smooth',
                          'slope_end_year_VH_L', 'slope_start_year_VH_L',
                          'slope_end_year_VH_L_smooth', 'slope_start_year_VH_L_smooth']]
Y_train = features_train['type']

X_test = features_est[['slope_end_year_VV_L', 'slope_start_year_VV_L',
                       'slope_end_year_VV_L_smooth', 'slope_start_year_VV_L_smooth',
                       'slope_end_year_VH_L', 'slope_start_year_VH_L',
                       'slope_end_year_VH_L_smooth', 'slope_start_year_VH_L_smooth']]

model.fit(X_train, Y_train)
features_est['pred'] = model.predict(X_test)

# read the shape file
gdf = gpd.read_file(shape)
gdf = gdf.merge(features_est, left_on='label', right_on='label', how='inner')
gdf.to_file(driver = 'ESRI Shapefile', filename= outputs/"result.shp")
# results.append(f1_score(y_test, y_pred))
