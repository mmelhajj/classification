from RF_classifier.correct_class_name import update_calss_name
from fmt.common import get_data_from_shape
from info import vect_clean_path

# get the shape file
shape = vect_clean_path / 'hand_map.shp'

gdf = get_data_from_shape(shape)

gdf['type_1st_h'] = gdf.apply(
    lambda row: update_calss_name[row['name']]['class_1st_half'] if row['name'] in update_calss_name.keys() else
    row['type_1st_h'], axis=1)

gdf['type_2nd_h'] = gdf.apply(
    lambda row: update_calss_name[row['name']]['class_2nd_half'] if row['name'] in update_calss_name.keys() else
    row['type_2nd_h'], axis=1)

# replace sudanese_corn by corn
gdf["type_1st_h"].replace({"sudanese_corn": "corn", "alfaalfa": "alfalfa"}, inplace=True)
gdf["type_2nd_h"].replace({"sudanese_corn": "corn", "alfaalfa": "alfalfa"}, inplace=True)

gdf.to_file(vect_clean_path / 'hand_map_fmt.shp')
