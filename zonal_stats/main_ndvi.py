import os
from pathlib import Path

import pandas as pd

from info import outputs, L8_path_clean, vect_clean_path
from zonal_stats.zonal_stat import mean_and_count_zonal_stat

shape = vect_clean_path / 'hand_map.shp'

# images
raster_files = Path(f"{L8_path_clean}").glob('*.tif')
dfs = []
for raster in raster_files:
    print(raster)
    stat = mean_and_count_zonal_stat(shape, raster, None, [1], date_string_start=12, date_string_end=20)
    dfs.append(pd.DataFrame(stat))

df = pd.concat(dfs)

# update col name
df.rename(columns={'band_1': 'ndvi'}, inplace=True)

# add date, and year
df['image_date'] = df['image_date_time_utc'].dt.date
df['year'] = df['image_date_time_utc'].dt.year

# add locale date time
df['image_date_time_ksa'] = df['image_date_time_utc'] + pd.to_timedelta(3, unit='h')

out = Path(outputs) / 'stats/'
if not os.path.exists(out):
    os.makedirs(out)

df.to_csv(f"{out}/hand_map_ndvi.csv", index=False)
