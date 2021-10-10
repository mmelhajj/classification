import joblib
import pandas as pd

from RF_classifier.example import get_features
from info import classfier, vect_clean_path
from mapping.common import get_data_from_shape

# get the shape file
shape = vect_clean_path / 'hand_map.shp'
gdf = get_data_from_shape(shape, ['name', 'geometry'])

# simulate
# get feature example, and set predictors
features, _, var_names = get_features(delete_multi_crop=True)
df_x = features[var_names]

# load the model and predict
model = joblib.load(f"{classfier}")
y_pred = model.predict(df_x)

# save pred in a shape file
df_pred = pd.DataFrame(y_pred, columns=['type_1st_h', 'type_2nd_h'])
df_pred['label'] = features['label']

# map
gdf = gdf.merge(df_pred, left_on='name', right_on='label', how='inner')
gdf.to_file('./map.shp')
