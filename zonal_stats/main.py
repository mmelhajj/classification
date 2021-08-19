import os
from pathlib import Path

import pandas as pd

from info import outputs, sar_path
from zonal_stats.zonal_stat_sar import mean_and_count_zonal_stat

# path to inputs
shape = Path(f"{outputs}/seg_rad50_sp20.shp")

# images
raster_files = Path(f"{sar_path}").glob('*.tif')
dfs = []
for raster in raster_files:
    print(raster)
    stat = mean_and_count_zonal_stat(shape, raster, None, [1, 2, 5])
    dfs.append(pd.DataFrame(stat))

df = pd.concat(dfs)

# update col name
df.rename(columns={'band_1': 'VV_L',
                   'band_2': 'VH_L',
                   'band_5': 'projectedInc'}, inplace=True)

# add incidence level
df['inc_class'] = df.apply(lambda row: 'high' if row['projectedInc'] > 40 else 'meduim', axis=1)

# add date, and year
df['image_date'] = df['image_date_time_utc'].dt.date
df['year'] = df['image_date_time_utc'].dt.year

# add locale date time
df['image_date_time_ksa'] = df['image_date_time_utc'] + pd.to_timedelta(3, unit='h')

out = Path(outputs) / 'stats/'
if not os.path.exists(out):
    os.makedirs(out)

df.to_csv(f"{out}/seg_rad50_sp20.csv", index=False)
